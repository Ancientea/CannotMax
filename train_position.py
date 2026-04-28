"""
基于位置顺序的训练脚本
数据格式：每个位置保存 (怪物ID, 数量) 而不是按ID聚合
"""
import time
from functools import cache
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from constants import POSITIONS_PER_SIDE, FEATURES_PER_POSITION
from recognize import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT


@cache
def get_device(prefer_gpu=True):
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    return torch.device("cpu")


device = get_device()
print(f"场地特征数量: {FIELD_FEATURE_COUNT}")


class ArknightsPositionDataset(Dataset):
    """
    位置顺序数据集
    数据格式：[左1_ID, 左1_数量, 左2_ID, 左2_数量, 左3_ID, 左3_数量,
             右1_ID, 右1_数量, 右2_ID, 右2_数量, 右3_ID, 右3_数量,
             场地特征L(6), 场地特征R(6), Result]
    """
    def __init__(self, csv_file, max_value=None):
        data = pd.read_csv(csv_file, header=None, skiprows=1)
        
        # 计算期望的列数
        position_features = POSITIONS_PER_SIDE * 2 * FEATURES_PER_POSITION  # 12
        field_features = FIELD_FEATURE_COUNT * 2  # 12
        expected_columns = position_features + field_features + 1  # +1 for Result
        
        if data.shape[1] < expected_columns:
            raise Exception(f"数据格式不符！期望至少 {expected_columns} 列，实际 {data.shape[1]} 列")
        
        # 提取特征和标签
        features = data.iloc[:, :expected_columns-1].values.astype(np.float32)
        labels = data.iloc[:, expected_columns-1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)
        
        # 分离位置特征和场地特征
        # 位置特征：[左1_ID, 左1_数量, ..., 右3_数量]
        position_end = position_features
        position_data = features[:, :position_end]
        
        # 场地特征
        field_data = features[:, position_end:position_end+field_features]
        
        # 分离左右两侧的位置数据
        left_positions = position_data[:, :POSITIONS_PER_SIDE*FEATURES_PER_POSITION]  # 6列
        right_positions = position_data[:, POSITIONS_PER_SIDE*FEATURES_PER_POSITION:]  # 6列
        
        # 分离左右场地特征
        left_field = field_data[:, :FIELD_FEATURE_COUNT]
        right_field = field_data[:, FIELD_FEATURE_COUNT:]
        
        # 提取ID和数量
        # 左侧：[ID1, 数量1, ID2, 数量2, ID3, 数量3]
        left_ids = left_positions[:, 0::2]  # 偶数索引
        left_counts = left_positions[:, 1::2]  # 奇数索引
        right_ids = right_positions[:, 0::2]
        right_counts = right_positions[:, 1::2]
        
        if max_value is not None:
            left_counts = np.clip(left_counts, 0, max_value)
            right_counts = np.clip(right_counts, 0, max_value)
        
        # 转换为PyTorch张量
        self.left_ids = torch.from_numpy(left_ids).long().to(device)
        self.left_counts = torch.from_numpy(left_counts).float().to(device)
        self.right_ids = torch.from_numpy(right_ids).long().to(device)
        self.right_counts = torch.from_numpy(right_counts).float().to(device)
        self.left_field = torch.from_numpy(left_field).float().to(device)
        self.right_field = torch.from_numpy(right_field).float().to(device)
        self.labels = torch.from_numpy(labels).float().to(device)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.left_ids[idx],
            self.left_counts[idx],
            self.right_ids[idx],
            self.right_counts[idx],
            self.left_field[idx],
            self.right_field[idx],
            self.labels[idx],
        )


class PositionAwareTransformer(nn.Module):
    """
    位置感知的Transformer模型
    考虑怪物的出场位置顺序
    """
    def __init__(self, num_monsters, num_field_features, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_monsters = num_monsters
        self.num_field_features = num_field_features
        self.embed_dim = embed_dim
        self.positions_per_side = POSITIONS_PER_SIDE
        
        # 怪物ID嵌入
        self.monster_embed = nn.Embedding(num_monsters + 1, embed_dim, padding_idx=0)  # +1 for padding
        nn.init.normal_(self.monster_embed.weight, mean=0.0, std=0.02)
        
        # 位置嵌入（左1, 左2, 左3, 右1, 右2, 右3）
        self.position_embed = nn.Embedding(6, embed_dim)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        
        # 场地特征处理
        self.field_proj = nn.Linear(num_field_features, embed_dim)
        
        # 数量特征融合
        self.count_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Transformer层
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.enemy_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.friend_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )
            self.norms.append(nn.LayerNorm(embed_dim))
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1)
        )
    
    def forward(self, left_ids, left_counts, right_ids, right_counts, left_field, right_field):
        batch_size = left_ids.shape[0]
        
        # 怪物嵌入 (B, 3, embed_dim)
        left_monster_embed = self.monster_embed(left_ids)
        right_monster_embed = self.monster_embed(right_ids)
        
        # 位置嵌入
        left_pos_ids = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)
        right_pos_ids = torch.arange(3, 6, device=device).unsqueeze(0).expand(batch_size, -1)
        left_pos_embed = self.position_embed(left_pos_ids)
        right_pos_embed = self.position_embed(right_pos_ids)
        
        # 融合怪物、位置和数量信息
        left_feat = left_monster_embed + left_pos_embed
        right_feat = right_monster_embed + right_pos_embed
        
        # 数量信息融合（前半维不变，后半维乘以数量）
        left_feat = torch.cat([
            left_feat[..., :self.embed_dim//2],
            left_feat[..., self.embed_dim//2:] * left_counts.unsqueeze(-1)
        ], dim=-1)
        right_feat = torch.cat([
            right_feat[..., :self.embed_dim//2],
            right_feat[..., self.embed_dim//2:] * right_counts.unsqueeze(-1)
        ], dim=-1)
        
        # 场地特征处理 (B, embed_dim)
        left_field_feat = self.field_proj(left_field).unsqueeze(1)  # (B, 1, embed_dim)
        right_field_feat = self.field_proj(right_field).unsqueeze(1)
        
        # 拼接场地特征到序列中 (B, 4, embed_dim)
        left_feat = torch.cat([left_feat, left_field_feat], dim=1)
        right_feat = torch.cat([right_feat, right_field_feat], dim=1)
        
        # 生成mask（ID为0的位置mask掉）
        left_mask = (left_ids == 0)  # (B, 3)
        right_mask = (right_ids == 0)
        # 场地特征不mask
        left_mask = torch.cat([left_mask, torch.zeros(batch_size, 1, device=device, dtype=torch.bool)], dim=1)
        right_mask = torch.cat([right_mask, torch.zeros(batch_size, 1, device=device, dtype=torch.bool)], dim=1)
        
        # Transformer层
        for i in range(len(self.enemy_attentions)):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=right_mask,
                need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=left_mask,
                need_weights=False,
            )
            
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)
            
            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=left_mask,
                need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=right_mask,
                need_weights=False,
            )
            
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)
            left_feat = self.norms[i](left_feat)
            right_feat = self.norms[i](right_feat)
        
        # 输出战斗力（对所有位置求和，mask的位置会被置0）
        left_power = self.fc(left_feat).squeeze(-1)  # (B, 4)
        right_power = self.fc(right_feat).squeeze(-1)
        
        # Mask掉无效位置
        left_power = left_power.masked_fill(left_mask, 0)
        right_power = right_power.masked_fill(right_mask, 0)
        
        # 计算总战斗力
        L = left_power.sum(1)
        R = right_power.sum(1)
        
        # 输出概率
        output = torch.sigmoid(R - L)
        return output


def train_one_epoch(model, train_loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for left_ids, left_counts, right_ids, right_counts, left_field, right_field, labels in train_loader:
        optimizer.zero_grad()
        
        try:
            with torch.amp.autocast_mode.autocast(
                device_type=device.type, enabled=(scaler is not None)
            ):
                outputs = model(left_ids, left_counts, right_ids, right_counts, left_field, right_field).squeeze()
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("警告: 模型输出包含NaN或Inf，跳过该批次")
                    continue
                
                outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                continue
            
            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        except RuntimeError as e:
            print(f"警告: 训练过程中出错 - {str(e)}")
            continue
    
    return total_loss / max(1, len(train_loader)), 100 * correct / max(1, total)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for left_ids, left_counts, right_ids, right_counts, left_field, right_field, labels in data_loader:
            try:
                with torch.amp.autocast_mode.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    outputs = model(left_ids, left_counts, right_ids, right_counts, left_field, right_field).squeeze()
                    
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        continue
                    
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    loss = criterion(outputs, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            except RuntimeError as e:
                print(f"警告: 评估过程中出错 - {str(e)}")
                continue
    
    return total_loss / max(1, len(data_loader)), 100 * correct / max(1, total)


def stratified_random_split(dataset, test_size=0.1, seed=42):
    labels = dataset.labels
    if str(device) != "cpu":
        labels = labels.cpu()
    labels = labels.numpy()
    
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )


def main():
    config = {
        "data_file": "arknights_position.csv",  # 新的位置顺序数据文件
        "batch_size": 1024,
        "test_size": 0.1,
        "embed_dim": 128,
        "n_layers": 3,
        "num_heads": 16,
        "lr": 3e-4,
        "epochs": 200,
        "seed": 42,
        "save_dir": "models",
        "max_feature_value": 100,
        "num_workers": 0 if torch.cuda.is_available() else 0,
    }
    
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    
    print(f"使用设备: {device}")
    
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.grad_scaler.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.amp.grad_scaler.GradScaler()
        print("CUDA可用，已启用混合精度训练")
    
    # 加载数据集
    dataset = ArknightsPositionDataset(
        config["data_file"],
        max_value=config["max_feature_value"],
    )
    
    train_dataset, val_dataset = stratified_random_split(
        dataset, test_size=config["test_size"], seed=config["seed"]
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    
    # 初始化模型
    model = PositionAwareTransformer(
        num_monsters=MONSTER_COUNT,
        num_field_features=FIELD_FEATURE_COUNT,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["n_layers"],
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    best_acc = 0
    best_loss = float("inf")
    
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, Path(config["save_dir"]) / "best_model_position_acc.pth")
            print("保存了新的最佳准确率模型!")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, Path(config["save_dir"]) / "best_model_position_loss.pth")
            print("保存了新的最佳损失模型!")
        
        torch.save(model, Path(config["save_dir"]) / "best_model_position_full.pth")
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-" * 40)
    
    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%, 最佳验证损失: {best_loss:.4f}")
    
    # 重命名模型文件
    current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_filename = f"position_data{len(dataset)}_acc{best_acc:.4f}_loss{best_loss:.4f}_{current_time_str}.pth"
    
    save_dir_path = Path(config["save_dir"])
    
    for old_name, new_prefix in [
        ("best_model_position_acc.pth", "best_model_position_acc_"),
        ("best_model_position_loss.pth", "best_model_position_loss_"),
        ("best_model_position_full.pth", "best_model_position_full_"),
    ]:
        old_path = save_dir_path / old_name
        new_path = save_dir_path / f"{new_prefix}{base_filename}"
        if old_path.exists():
            old_path.rename(new_path)
            print(f"模型文件已重命名: {old_path} -> {new_path}")


if __name__ == "__main__":
    main()
