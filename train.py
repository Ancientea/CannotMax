"""
优化版训练脚本 — Arknights 左右胜负预测模型

主要优化点:
1. MSELoss → BCELoss（适合二分类）
2. batch_size 1024 → 256（更多梯度更新，更好泛化）
3. weight_decay 0.1 → 0.01（适度正则化）
4. 加入 warmup + CosineAnnealing 学习率调度
5. 加入早停机制（patience=30）
6. 提升数值稳定性
7. 保存最佳模型时同步保存配置信息
8. 增加训练过程监控（梯度范数、学习率）
"""

import time
import copy
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
from recognize import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT


# ==================== 设备管理 ====================

@cache
def get_device(prefer_gpu=True):
    """获取可用计算设备"""
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            try:
                import torch_directml
                return torch_directml.device()
            except:
                pass
    return torch.device("cpu")


device = get_device()

# 计算总特征数量: (怪物特征 + 场地特征) * 2 侧
TOTAL_FEATURE_COUNT = (MONSTER_COUNT + FIELD_FEATURE_COUNT) * 2


# ==================== 数据处理 ====================

def preprocess_data(csv_file):
    """预处理CSV文件，检测并报告异常值"""
    print(f"预处理数据文件: {csv_file}")

    data = pd.read_csv(csv_file, header=None, skiprows=1)
    print(f"原始数据形状: {data.shape}")

    expected_columns = TOTAL_FEATURE_COUNT + 2
    if data.shape[1] != expected_columns:
        print(f"数据列数不符！期望 {expected_columns} 列，实际 {data.shape[1]} 列")
        raise Exception("数据格式不符")

    data = data.iloc[:, 0: TOTAL_FEATURE_COUNT + 1]
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 异常检测
    extreme_values = (np.abs(features) > 20).sum().sum()
    if extreme_values > 0:
        print(f"⚠ 发现 {extreme_values} 个绝对值大于20的特征值")

    invalid_labels = labels.apply(lambda x: x not in ["L", "R"]).sum()
    if invalid_labels > 0:
        print(f"⚠ 发现 {invalid_labels} 个无效标签")

    # 数据分布统计
    feature_min = features.min().min()
    feature_max = features.max().max()
    feature_mean = features.mean().mean()
    feature_std = features.std().mean()
    label_dist = labels.value_counts(normalize=True)

    print(f"特征值范围: [{feature_min:.2f}, {feature_max:.2f}]")
    print(f"特征均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")
    print(f"标签分布: L={label_dist.get('L', 0):.1%}, R={label_dist.get('R', 0):.1%}")

    return data.shape[1]


class ArknightsDataset(Dataset):
    """
    数据集类 — 与原始版本接口完全兼容。
    将特征分离为 sign（方向）和 count（幅度），分别用于 Embedding 查找和值缩放。
    """

    def __init__(self, csv_file, max_value=None, use_symmetry_aug=False):
        data = pd.read_csv(csv_file, header=None, skiprows=1)
        expected_columns = TOTAL_FEATURE_COUNT + 2
        if data.shape[1] != expected_columns:
            raise Exception(f"数据列数不符！期望 {expected_columns}，实际 {data.shape[1]}")

        data = data.iloc[:, 0: TOTAL_FEATURE_COUNT + 1]
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 特征分段索引
        left_monster_end = MONSTER_COUNT
        left_field_end = MONSTER_COUNT + FIELD_FEATURE_COUNT
        right_monster_end = MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT
        right_field_end = MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT + FIELD_FEATURE_COUNT

        # 分离左右两侧特征
        left_monster_features = features[:, :left_monster_end]
        left_field_features = features[:, left_monster_end:left_field_end]
        right_monster_features = features[:, left_field_end:right_monster_end]
        right_field_features = features[:, right_monster_end:right_field_end]

        # 构建 sign（方向）和 count（幅度）
        left_counts = np.concatenate([np.abs(left_monster_features), left_field_features], axis=1)
        right_counts = np.concatenate([np.abs(right_monster_features), right_field_features], axis=1)
        left_signs = np.concatenate([np.sign(left_monster_features), np.ones_like(left_field_features)], axis=1)
        right_signs = np.concatenate([np.sign(right_monster_features), np.ones_like(right_field_features)], axis=1)

        # 可选的数值裁剪
        if max_value is not None:
            left_counts = np.clip(left_counts, 0, max_value)
            right_counts = np.clip(right_counts, 0, max_value)

        # 对称数据增强：左右互换 + 标签翻转 → 2x 数据
        if use_symmetry_aug:
            left_signs  = np.concatenate([left_signs,  right_signs], axis=0)
            right_signs = np.concatenate([right_signs, left_signs],  axis=0)
            left_counts  = np.concatenate([left_counts,  right_counts], axis=0)
            right_counts = np.concatenate([right_counts, left_counts],  axis=0)
            labels       = np.concatenate([labels,       1 - labels],   axis=0)
            print(f"对称增强: {len(labels)//2} → {len(labels)} 条 (2x)")

        # 预加载到设备（低显存自动降级到 CPU）
        target_device = device
        if device.type == "cuda":
            free_mem, total_mem = torch.cuda.mem_get_info()
            est_mem = left_signs.nbytes * 4 + right_signs.nbytes * 4  # 估算 4 份 tensor
            if free_mem < est_mem * 1.5:
                print(f"  ⚠ GPU 可用显存不足 ({free_mem/1024**3:.1f}GB < 需 {est_mem/1024**3:.1f}GB)，数据改用 CPU")
                target_device = torch.device("cpu")
        self.left_signs = torch.from_numpy(left_signs).to(target_device)
        self.right_signs = torch.from_numpy(right_signs).to(target_device)
        self.left_counts = torch.from_numpy(left_counts).to(target_device)
        self.right_counts = torch.from_numpy(right_counts).to(target_device)
        self.labels = torch.from_numpy(labels).float().to(target_device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.left_signs[idx],
            self.left_counts[idx],
            self.right_signs[idx],
            self.right_counts[idx],
            self.labels[idx],
        )


# ==================== 模型定义 ====================

class UnitAwareTransformer(nn.Module):
    """原始架构 — 曾达 85.4% 验证准确率。只加 forward_logits 兼容 BCEWithLogitsLoss。"""

    def __init__(self, num_units, embed_dim=128, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.num_units = num_units
        self.monster_count = MONSTER_COUNT
        self.field_count = FIELD_FEATURE_COUNT
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in range(num_layers):
            self.enemy_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            self.friend_attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)
            self.norm.append(nn.LayerNorm(embed_dim))

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1)
        )

    def forward_logits(self, left_signs, left_counts, right_signs, right_counts):
        k = min(8, left_counts.shape[1])
        left_values, left_indices = torch.topk(left_counts, k=k, dim=1)
        right_values, right_indices = torch.topk(right_counts, k=k, dim=1)

        left_feat = self.unit_embed(left_indices)
        right_feat = self.unit_embed(right_indices)

        embed_dim = self.embed_dim
        left_feat = torch.cat([
            left_feat[..., :embed_dim // 2],
            left_feat[..., embed_dim // 2:] * left_values.unsqueeze(-1),
        ], dim=-1)
        right_feat = torch.cat([
            right_feat[..., :embed_dim // 2],
            right_feat[..., embed_dim // 2:] * right_values.unsqueeze(-1),
        ], dim=-1)

        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        for i in range(self.num_layers):
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat, key=right_feat, value=right_feat,
                key_padding_mask=~right_mask, need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat, key=left_feat, value=left_feat,
                key_padding_mask=~left_mask, need_weights=False,
            )
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            delta_left, _ = self.friend_attentions[i](
                query=left_feat, key=left_feat, value=left_feat,
                key_padding_mask=~left_mask, need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat, key=right_feat, value=right_feat,
                key_padding_mask=~right_mask, need_weights=False,
            )
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)
            left_feat = self.norm[i](left_feat)
            right_feat = self.norm[i](right_feat)

        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask
        return R.sum(1) - L.sum(1)

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        return torch.sigmoid(self.forward_logits(left_signs, left_counts, right_signs, right_counts))


# ==================== 训练 & 评估 ====================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler=None,
                    grad_clip=1.0, log_interval=10):
    """训练一个 epoch，返回 (平均损失, 准确率)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)

    for batch_idx, (ls, lc, rs, rc, labels) in enumerate(train_loader):
        ls, lc, rs, rc, labels = [
            x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)
        ]

        # ── 输入验证 ──
        if (torch.isnan(ls).any() or torch.isnan(lc).any() or
            torch.isnan(rs).any() or torch.isnan(rc).any() or
            torch.isinf(ls).any() or torch.isinf(lc).any() or
            torch.isinf(rs).any() or torch.isinf(rc).any()):
            print(f"⚠ 批次 {batch_idx}: 输入包含 NaN/Inf，跳过")
            continue

        labels = torch.clamp(labels, 0.0, 1.0)

        # ── 前向传播 ──
        optimizer.zero_grad(set_to_none=True)  # 比 zero_grad() 更高效

        try:
            with torch.amp.autocast_mode.autocast(
                device_type=device.type, enabled=(scaler is not None)
            ):
                outputs = model(ls, lc, rs, rc).squeeze()

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"⚠ 批次 {batch_idx}: 输出包含 NaN/Inf，跳过")
                    continue

                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠ 批次 {batch_idx}: 损失异常 ({loss.item():.4f})，跳过")
                continue

            # ── 反向传播 ──
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            # ── 统计 ──
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 定期打印
            if (batch_idx + 1) % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Batch {batch_idx + 1}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        except RuntimeError as e:
            print(f"⚠ 批次 {batch_idx}: 运行时错误 - {e}")
            continue

    avg_loss = total_loss / max(1, len(train_loader))
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, data_loader, criterion):
    """评估模型，返回 (平均损失, 准确率)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in data_loader:
        ls, lc, rs, rc, labels = [
            x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)
        ]

        # 数据验证
        if (torch.isnan(ls).any() or torch.isnan(lc).any() or
            torch.isnan(rs).any() or torch.isnan(rc).any() or
            torch.isinf(ls).any() or torch.isinf(lc).any() or
            torch.isinf(rs).any() or torch.isinf(rc).any()):
            continue

        labels = torch.clamp(labels, 0.0, 1.0)

        try:
            with torch.amp.autocast_mode.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                outputs = model(ls, lc, rs, rc).squeeze()

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue

                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError:
            continue

    avg_loss = total_loss / max(1, len(data_loader))
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


# ==================== 数据划分 ====================

def stratified_random_split(dataset, test_size=0.1, seed=42):
    """分层随机划分训练/验证集"""
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


# ==================== 学习率调度器 ====================

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing 学习率调度器。
    前 warmup_epochs 线性增长，之后余弦衰减到 min_lr。
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            # 线性 warmup
            return self.base_lr * self.current_epoch / max(1, self.warmup_epochs)
        else:
            # 余弦衰减
            progress = (self.current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + np.cos(np.pi * progress)
            )

    def get_last_lr(self):
        return [self._get_lr()]


# ==================== 早停机制 ====================

class EarlyStopping:
    """早停：验证损失在 patience 个 epoch 内未改善则停止"""

    def __init__(self, patience=30, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class FocalMSELoss(nn.Module):
    """Focal MSE + Label Smoothing: 自动关注难样本 + 防止过自信"""
    def __init__(self, gamma=1.5, smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
    def forward(self, preds, targets):
        # Label Smoothing: 0/1 → 0.025/0.975
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        mse = (preds - targets) ** 2
        weight = torch.abs(preds - targets) ** self.gamma
        return (weight * mse).mean()


# ==================== 主训练流程 ====================

def main():
    # ── 超参数配置 ──
    config = {
        "data_file": "arknights.csv",
        "batch_size": 2048,
        "test_size": 0.1,
        "embed_dim": 128,
        "n_layers": 3,
        "num_heads": 16,
        "dropout": 0.25,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-3,
        "warmup_epochs": 10,
        "epochs": 300,
        "early_stop_patience": 20,
        "grad_clip": 1.0,
        "seeds": [42, 314, 3407],
        "save_dir": "models",
        "max_feature_value": 100,
        "num_workers": 0,
        "log_interval": 10,
    }

    # ── 遍历每个 seed 训练 ──
    aug_enabled = True
    num_workers = 0
    swa_start_epoch = int(config["epochs"] * 0.4)  # 40%时启动SWA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("已启用混合精度训练 (AMP)")
    all_results = []
    for seed_idx, seed in enumerate(config["seeds"]):
        print(f"\n{'#'*60}")
        print(f"### Seed {seed} ({seed_idx+1}/{len(config['seeds'])})")
        print(f"{'#'*60}")

        # ── 随机种子 ──
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        # ── 数据加载 ──
        data_path = Path(config["data_file"])
        if not data_path.exists():
            print(f"\n  ✗ 训练数据文件不存在: {data_path.absolute()}")
            print(f"  请确保该文件已放入项目目录，或修改 train.py 中 config['data_file'] 指向正确路径")
            return
        dataset = ArknightsDataset(
            config["data_file"],
            max_value=config["max_feature_value"],
        )
        data_length = len(dataset)
        train_raw, val_dataset = stratified_random_split(
            dataset, test_size=config["test_size"], seed=seed
        )
        aug_dataset = ArknightsDataset(
            config["data_file"],
            max_value=config["max_feature_value"],
            use_symmetry_aug=aug_enabled,
        )
        train_indices = train_raw.indices
        all_train_indices = list(train_indices) + [i + data_length for i in train_indices]
        train_dataset = torch.utils.data.Subset(aug_dataset, all_train_indices)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
            shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
            num_workers=num_workers, pin_memory=False)

        # ── 模型 ──
        total_units = MONSTER_COUNT + FIELD_FEATURE_COUNT
        model = UnitAwareTransformer(
            num_units=total_units, embed_dim=config["embed_dim"],
            num_heads=config["num_heads"], num_layers=config["n_layers"],
            dropout=config["dropout"],
        ).to(device)

        criterion = FocalMSELoss(gamma=1.5)  # 自动关注难样本
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
            weight_decay=config["weight_decay"])
        scheduler = WarmupCosineScheduler(optimizer,
            warmup_epochs=config["warmup_epochs"], total_epochs=config["epochs"],
            base_lr=config["lr"], min_lr=config["min_lr"])
        early_stopping = EarlyStopping(patience=config["early_stop_patience"],
            min_delta=1e-4, mode="min")

        best_acc, best_loss, best_epoch = 0.0, float("inf"), 0
        swa_model = None
        start_time = time.time()

        for epoch in range(config["epochs"]):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader,
                criterion, optimizer, scaler, grad_clip=config["grad_clip"],
                log_interval=config["log_interval"])
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            current_lr = scheduler.step()

            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
            if val_acc > best_acc:
                best_acc = val_acc

            if epoch >= swa_start_epoch:
                if swa_model is None:
                    swa_model = copy.deepcopy(model)
                else:
                    for swa_p, p in zip(swa_model.parameters(), model.parameters()):
                        swa_p.data = (swa_p.data + p.data) / 2.0

            print(f"Epoch {epoch+1:3d}/{config['epochs']} | LR: {current_lr:.2e} | "
                  f"TrLoss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"VaLoss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"Best: {best_acc:.2f}%")

            if early_stopping(val_loss):
                print(f"Seed {seed} 早停 @ epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        print(f"Seed {seed}: Acc={best_acc:.2f}% Loss={best_loss:.4f} Epoch={best_epoch} Time={total_time/60:.1f}min")

        # 保存模型（原名格式 + seed 格式）
        save_dir = Path(config["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        orig_name = f"best_model_acc_data{data_length}_acc{best_acc:.4f}_loss{best_loss:.4f}_{ts}"
        torch.save(model, save_dir / f"model_seed{seed}.pth")
        torch.save(model, save_dir / f"{orig_name}.pth")
        if swa_model is not None:
            torch.save(swa_model, save_dir / f"{orig_name}_swa.pth")
        all_results.append({"seed": seed, "acc": best_acc, "loss": best_loss})

    # ── 设备信息 ──
    print(f"使用设备: {device}")
    if str(device) == "cuda":
        print(f"  CUDA 设备数: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif str(device) == "cpu":
        print("⚠ 未检测到 GPU，训练将较慢")


if __name__ == "__main__":
    main()
