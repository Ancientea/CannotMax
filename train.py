import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
from torch.cuda.amp import GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data(csv_file):
    """预处理CSV文件，将异常值修正为合理范围"""
    print(f"预处理数据文件: {csv_file}")

    # 读取CSV文件
    data = pd.read_csv(csv_file, header=None, skiprows=1)
    print(f"原始数据形状: {data.shape}")

    # 检查特征范围
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 统计极端值
    extreme_values = (np.abs(features) > 20).sum().sum()
    if extreme_values > 0:
        print(f"发现 {extreme_values} 个绝对值大于20的特征值")

    # 检查标签
    invalid_labels = labels.apply(lambda x: x not in ["L", "R"]).sum()
    if invalid_labels > 0:
        print(f"发现 {invalid_labels} 个无效标签")

    # 输出特征的范围信息
    feature_min = features.min().min()
    feature_max = features.max().max()
    feature_mean = features.mean().mean()
    feature_std = features.std().mean()

    print(f"特征值范围: [{feature_min}, {feature_max}]")
    print(f"特征值平均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")

    # 如果需要，可以在这里对数据进行更多的预处理
    # 例如：将极端值截断到合理范围

    return data.shape[1]


class ArknightsDataset(Dataset):
    def __init__(self, csv_file, max_value=None):
        data = pd.read_csv(csv_file, header=None, skiprows=1)
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分割双方单位
        feature_count = features.shape[1]
        midpoint = feature_count // 2
        left_counts = np.abs(features[:, :midpoint])
        right_counts = np.abs(features[:, midpoint:])
        left_signs = np.sign(features[:, :midpoint])
        right_signs = np.sign(features[:, midpoint:])

        if max_value is not None:
            left_counts = np.clip(left_counts, 0, max_value)
            right_counts = np.clip(right_counts, 0, max_value)

        # 转换为 PyTorch 张量，并一次性加载到 GPU
        self.left_signs = torch.from_numpy(left_signs).to(device)
        self.right_signs = torch.from_numpy(right_signs).to(device)
        self.left_counts = torch.from_numpy(left_counts).to(device)
        self.right_counts = torch.from_numpy(right_counts).to(device)
        self.labels = torch.from_numpy(labels).float().to(device)

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


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=0.2
                )
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=0.2
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

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, left_sign, left_count, right_sign, right_count):
        # 提取Top3兵种特征
        left_values, left_indices = torch.topk(left_count, k=3, dim=1)
        right_values, right_indices = torch.topk(right_count, k=3, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, 3, 128)
        right_feat = self.unit_embed(right_indices)  # (B, 3, 128)

        embed_dim = self.embed_dim

        # 前x维不变，后y维 *= 数量，但使用缩放后的值
        left_feat = torch.cat(
            [
                left_feat[..., : embed_dim // 2],  # 前x维
                left_feat[..., embed_dim // 2 :]
                * left_values.unsqueeze(-1),  # 后y维乘数量
            ],
            dim=-1,
        )
        right_feat = torch.cat(
            [
                right_feat[..., : embed_dim // 2],
                right_feat[..., embed_dim // 2 :] * right_values.unsqueeze(-1),
            ],
            dim=-1,
        )

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, 3) 0.1防一手可能的浮点误差
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)

        # 输出战斗力
        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask

        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R.sum(1) - L.sum(1))

        return output


def train_one_epoch(model, train_loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)]

        optimizer.zero_grad()

        # 检查输入值范围
        if (
            torch.isnan(ls).any()
            or torch.isnan(lc).any()
            or torch.isnan(rs).any()
            or torch.isnan(rc).any()
        ):
            print("警告: 输入数据包含NaN，跳过该批次")
            continue

        if (
            torch.isinf(ls).any()
            or torch.isinf(lc).any()
            or torch.isinf(rs).any()
            or torch.isinf(rc).any()
        ):
            print("警告: 输入数据包含Inf，跳过该批次")
            continue

        # 确保labels严格在0-1之间
        if (labels < 0).any() or (labels > 1).any():
            print("警告: 标签值不在[0,1]范围内，进行修正")
            labels = torch.clamp(labels, 0, 1)

        try:
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(ls, lc, rs, rc).squeeze()
                # 确保输出在合理范围内
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("警告: 模型输出包含NaN或Inf，跳过该批次")
                    continue

                # 确保输出严格在0-1之间，因为BCELoss需要
                if (outputs < 0).any() or (outputs > 1).any():
                    print("警告: 模型输出不在[0,1]范围内，进行修正")
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs, labels)

            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                continue

            if scaler:  # 使用混合精度
                scaler.scale(loss).backward()
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # 不使用混合精度
                loss.backward()
                # 梯度裁剪，避免梯度爆炸
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
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)]

            # 检查输入值范围
            if (
                torch.isnan(ls).any()
                or torch.isnan(lc).any()
                or torch.isnan(rs).any()
                or torch.isnan(rc).any()
                or torch.isinf(ls).any()
                or torch.isinf(lc).any()
                or torch.isinf(rs).any()
                or torch.isinf(rc).any()
            ):
                print("警告: 评估时输入数据包含NaN或Inf，跳过该批次")
                continue

            # 确保labels严格在0-1之间
            if (labels < 0).any() or (labels > 1).any():
                labels = torch.clamp(labels, 0, 1)

            try:
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(ls, lc, rs, rc).squeeze()
                    # 确保输出在合理范围内
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print("警告: 评估时模型输出包含NaN或Inf，跳过该批次")
                        continue

                    # 确保输出严格在0-1之间，因为BCELoss需要
                    if (outputs < 0).any() or (outputs > 1).any():
                        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs, labels)

                # 检查loss是否有效
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
    labels = dataset.labels  # 假设 labels 是一个 GPU tensor
    if device != "cpu":
        labels = labels.cpu()  # 移动到 CPU 上进行操作
    labels = labels.numpy()  # 转换为 numpy array

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )


def main():
    # 配置参数
    config = {
        "data_file": "arknights.csv",
        "batch_size": 2048,  # 128/512/2048
        "test_size": 0.1,
        "embed_dim": 256,  # 128可能不够，512会过拟合
        "n_layers": 4,  # 3也可以
        "num_heads": 8,
        "lr": 5e-4,  # 3e-4
        "epochs": 100,  # 一般30也够了
        "seed": 42,  # 随机数种子
        "save_dir": "models",  # 存到哪里
        "max_feature_value": 100,  # 限制特征最大值，防止极端值造成不稳定
        "num_workers": 0 if torch.cuda.is_available() else 0,  # 根据CUDA可用性设置num_workers
    }

    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # 设置设备
    print(f"使用设备: {device}")

    # 初始化 GradScaler 用于混合精度训练
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = GradScaler() # 如果是老版本
        print("CUDA可用，已启用混合精度训练的GradScaler。")

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

        # 设置确定性计算以增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        print("警告: 未检测到GPU，将在CPU上运行训练，这可能会很慢!")

    # 先预处理数据，检查是否有异常值
    num_data = preprocess_data(config["data_file"])

    # 加载数据集
    dataset = ArknightsDataset(
        config["data_file"], max_value=config["max_feature_value"]  # 使用最大值限制
    )

    # 数据集分割
    val_size = int(0.1 * len(dataset))  # 10% 验证集
    train_size = len(dataset) - val_size

    # 划分
    train_dataset, val_dataset = stratified_random_split(
        dataset, test_size=config["test_size"], seed=config["seed"]
    )

    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # 初始化模型
    model = UnitAwareTransformer(
        num_units=(num_data - 1) // 2,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["n_layers"],
    ).to(device)

    print(
        f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 训练设置
    best_acc = 0
    best_loss = float("inf")

    # 训练循环
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model,
                os.path.join(config["save_dir"], "best_model_acc.pth"),
            )

            print("保存了新的最佳准确率模型!")
        else:
            print(f"最佳准确率为: {best_acc:.2f}")

        # 保存最佳模型（基于损失）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model,
                os.path.join(config["save_dir"], "best_model_loss.pth"),
            )
            print("保存了新的最佳损失模型!")
        else:
            print(f"最佳损失为: {best_loss:.4f}")

        torch.save(model, os.path.join(config["save_dir"], "best_model_full.pth")) # 最后一次计算的模型

        # 保存最新模型
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        #     'train_acc': train_acc,
        #     'val_acc': val_acc,
        #     'config': config
        # }, os.path.join(config['save_dir'], 'latest_checkpoint.pth'))

        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-" * 40)

        # 计时
        if epoch == 0:
            start_time = time.time()
            epoch_start_time = start_time
        else:
            current_time = time.time()
            epoch_duration = current_time - epoch_start_time
            elapsed_time = current_time - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            estimated_total_time = avg_epoch_time * config["epochs"]
            remaining_time = estimated_total_time - elapsed_time

            print(f"Epoch Time: {epoch_duration:.2f}s")
            print(f"Elapsed Time: {elapsed_time / 60:.2f}min")
            print(f"Estimated Remaining Time: {remaining_time / 60:.2f}min")
            print(f"Estimated Total Time: {estimated_total_time / 60:.2f}min")
            epoch_start_time = current_time  # Reset for next epoch

        print("-" * 40)

        # 绘制并保存训练历史
        # if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
        #     plot_training_history(
        #         train_losses, val_losses, train_accs, val_accs,
        #         save_path=os.path.join(config['save_dir'], 'training_history.png')
        #     )

    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%, 最佳验证损失: {best_loss:.4f}")

    # 保存最终训练历史
    # plot_training_history(
    #     train_losses, val_losses, train_accs, val_accs,
    #     save_path=os.path.join(config['save_dir'], 'final_training_history.png')
    # )


if __name__ == "__main__":
    main()
