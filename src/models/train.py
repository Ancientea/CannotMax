import time
from functools import cache
from datetime import datetime
from pathlib import Path
import sys
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Handle both direct execution and module import
if __name__ == "__main__":
    # Direct execution: add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
    from src.models.model import UnitAwareTransformer
    from src.models.muon import get_muon_lion_optimizers
else:
    # Module import
    from ..core.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
    from .model import UnitAwareTransformer
    from .muon import get_muon_lion_optimizers

print(f"场地特征数量: {FIELD_FEATURE_COUNT}")

# 计算总特征数量 (怪物特征 + 场地特征) * 2 + Result + ImgPath
TOTAL_FEATURE_COUNT = (MONSTER_COUNT + FIELD_FEATURE_COUNT) * 2


@cache
def get_device(prefer_gpu=True):
    """
    prefer_gpu (bool): 是否优先尝试使用GPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")  # Apple Silicon GPU
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
            return torch.device("xpu")
    return torch.device("cpu")


device = get_device()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_file(csv_file: str) -> Path:
    """解析训练数据路径，支持相对路径和常见回退路径。"""
    requested = Path(csv_file)

    candidates = []
    # 优先：用户显式传入的路径（绝对或当前工作目录相对）
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(requested)
        # 再尝试以项目根目录为基准
        candidates.append(PROJECT_ROOT / requested)

    # 默认回退候选（兼容重构后目录）
    candidates.extend(
        [
            PROJECT_ROOT / "arknights.csv",
            PROJECT_ROOT / "data" / "train" / "arknights.csv",
            PROJECT_ROOT / "data" / "arknights.csv",
            PROJECT_ROOT / "src" / "simulation" / "arknights.csv",
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file():
            resolved = path.resolve()
            if Path(csv_file) != resolved:
                print(f"训练数据路径已解析为: {resolved}")
            return resolved

    unique_candidates = []
    seen = set()
    for path in candidates:
        text = str(path)
        if text not in seen:
            seen.add(text)
            unique_candidates.append(text)

    raise FileNotFoundError(
        "未找到训练数据文件，请检查路径。已尝试以下位置:\n"
        + "\n".join(f"- {p}" for p in unique_candidates)
    )


def plot_learning_curve(
    train_losses, val_losses, train_accs, val_accs, save_path
):
    """绘制学习曲线并保存为图片"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    plt.plot(epochs, val_losses, "r-", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "b-", label="Train Acc")
    plt.plot(epochs, val_accs, "r-", label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"学习曲线已保存至: {save_path}")
    plt.close()


def preprocess_data(csv_file):
    """预处理CSV文件，将异常值修正为合理范围"""
    csv_path = resolve_data_file(csv_file)
    print(f"预处理数据文件: {csv_path}")

    # 读取CSV文件
    data = pd.read_csv(csv_path, header=None, skiprows=1)
    print(f"原始数据形状: {data.shape}")

    # 检查数据形状
    expected_columns = TOTAL_FEATURE_COUNT + 2  # +2 for Result and ImgPath
    if data.shape[1] != expected_columns:
        print(
            f"数据列数不符！期望 {expected_columns} 列，实际 {data.shape[1]} 列"
        )
        print(
            f"期望格式: {MONSTER_COUNT}(怪物L) + {FIELD_FEATURE_COUNT}(场地L) + {MONSTER_COUNT}(怪物R) + {FIELD_FEATURE_COUNT}(场地R) + 1(Result) + 1(ImgPath)"
        )
        raise Exception("数据格式不符")

    data = data.iloc[
        :, 0 : TOTAL_FEATURE_COUNT + 1
    ]  # 保留特征和结果列，去掉ImgPath

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

    return data.shape[1]


class ArknightsDataset(Dataset):
    def __init__(self, csv_file, max_value=None):
        csv_path = resolve_data_file(csv_file)
        data = pd.read_csv(csv_path, header=None, skiprows=1)
        # 检查数据形状
        expected_columns = TOTAL_FEATURE_COUNT + 2  # +2 for Result and ImgPath
        if data.shape[1] != expected_columns:
            print(
                f"数据列数不符！期望 {expected_columns} 列，实际 {data.shape[1]} 列"
            )
            raise Exception("数据格式不符")
        data = data.iloc[
            :, 0 : TOTAL_FEATURE_COUNT + 1
        ]  # 保留特征和结果列，去掉ImgPath
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(
            np.float32
        )

        # 分割双方单位和场地特征
        # 数据格式: [怪物L(77), 场地L(6), 怪物R(77), 场地R(6)]
        left_monster_end = MONSTER_COUNT
        left_field_end = MONSTER_COUNT + FIELD_FEATURE_COUNT
        right_monster_end = MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT
        right_field_end = (
            MONSTER_COUNT
            + FIELD_FEATURE_COUNT
            + MONSTER_COUNT
            + FIELD_FEATURE_COUNT
        )

        # 提取各部分特征
        left_monster_features = features[:, :left_monster_end]
        left_field_features = features[:, left_monster_end:left_field_end]
        right_monster_features = features[:, left_field_end:right_monster_end]
        right_field_features = features[:, right_monster_end:right_field_end]

        # 合并怪物特征和场地特征（场地特征直接使用，不取绝对值和符号）
        left_counts = np.concatenate(
            [np.abs(left_monster_features), left_field_features], axis=1
        )
        right_counts = np.concatenate(
            [np.abs(right_monster_features), right_field_features], axis=1
        )
        left_signs = np.concatenate(
            [
                np.sign(left_monster_features),
                np.ones_like(left_field_features),
            ],
            axis=1,
        )
        right_signs = np.concatenate(
            [
                np.sign(right_monster_features),
                np.ones_like(right_field_features),
            ],
            axis=1,
        )

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


def train_one_epoch(
    model, train_loader, criterion, muon_opt, lion_opt, scaler=None
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [
            x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)
        ]

        # 清空所有的梯度
        muon_opt.zero_grad()
        lion_opt.zero_grad()

        # 检查输入数据
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
            with torch.amp.autocast_mode.autocast(
                device_type=device.type, enabled=(scaler is not None)
            ):
                outputs = model(ls, lc, rs, rc).squeeze()
                # 确保输出在合理范围内
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("警告: 模型输出包含NaN或Inf，跳过该批次")
                    continue

                # 确保输出严格在0-1之间
                if (outputs < 0).any() or (outputs > 1).any():
                    print("警告: 模型输出不在[0,1]范围内，进行修正")
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

            # 将损失函数计算移出 autocast 区域，并强制使用 float32 避免安全警告
            loss = criterion(outputs.float(), labels.float())

            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                continue

            if scaler:  # 使用混合精度
                scaler.scale(loss).backward()
                # 混合优化器时需要分别 unscale 进行统一梯度裁剪
                scaler.unscale_(muon_opt)
                scaler.unscale_(lion_opt)
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(muon_opt)
                scaler.step(lion_opt)
                scaler.update()
            else:  # 不使用混合精度
                loss.backward()
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                muon_opt.step()
                lion_opt.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            print(f"警告: 训练过程中出错 - {str(e)}")
            continue

    return total_loss / max(1, len(train_loader)), 100 * correct / max(
        1, total
    )


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [
                x.to(device, non_blocking=True)
                for x in (ls, lc, rs, rc, labels)
            ]

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
                with torch.amp.autocast_mode.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    outputs = model(ls, lc, rs, rc).squeeze()
                    # 确保输出在合理范围内
                    if (
                        torch.isnan(outputs).any()
                        or torch.isinf(outputs).any()
                    ):
                        print("警告: 评估时模型输出包含NaN或Inf，跳过该批次")
                        continue
                    # 确保输出严格在0-1之间
                    if (outputs < 0).any() or (outputs > 1).any():
                        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                # 将损失函数计算移出 autocast 区域，并强制使用 float32 避免安全警告
                loss = criterion(outputs.float(), labels.float())

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
    if str(device) != "cpu":
        labels = labels.cpu()  # 移动到 CPU 上进行操作
    labels = labels.numpy()  # 转换为 numpy array

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
        "data_file": "data/arknights.csv",
        "batch_size": 1024,
        "test_size": 0.1,
        "embed_dim": 256,
        "n_layers": 3,
        "num_heads": 4,
        "dropout": 0.3,  # Dropout 设置
        "lr": 3e-4,  # 新优化器可以改大一点
        "lion_lr": 3e-4 / 10,  # 论文指出 Lion 优化器需要更小的学习率
        "epochs": 50,
        "seed": 42,  # 随机数种子
        "save_dir": PROJECT_ROOT / "models",  # 存到哪里
        "max_feature_value": 100,  # 限制特征最大值，防止极端值造成不稳定
        "num_workers": (
            0 if torch.cuda.is_available() else 0
        ),  # 根据CUDA可用性设置num_workers
    }

    # 创建保存目录
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # 设置设备
    print(f"使用设备: {device}")

    # 初始化 GradScaler
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.grad_scaler.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.amp.grad_scaler.GradScaler()  # 如果是老版本
        print("CUDA可用，已启用混合精度训练的GradScaler。")

    # 检查CUDA可用性
    if str(device) == "cuda":
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        # 设置确定性计算以增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif str(device) == "cpu":
        print("警告: 未检测到GPU，将在CPU上运行训练!")

    # 先预处理数据，检查是否有异常值
    num_data = preprocess_data(config["data_file"])

    # 加载数据集
    dataset = ArknightsDataset(
        config["data_file"],
        max_value=config["max_feature_value"],  # 使用最大值限制
    )

    # 数据集分割
    data_length = len(dataset)
    val_size = int(0.1 * data_length)  # 10% 验证集
    train_size = data_length - val_size

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
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # 初始化模型
    # num_units 现在包括怪物数量和场地特征数量
    total_units = MONSTER_COUNT + FIELD_FEATURE_COUNT
    model = UnitAwareTransformer(
        num_units=total_units,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["n_layers"],
        dropout=config["dropout"],  # 传入 dropout
    ).to(device)

    print(
        f"模型使用特征数: 怪物({MONSTER_COUNT}) + 场地({FIELD_FEATURE_COUNT}) = {total_units}"
    )
    print(
        f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # 损失函数和优化器 (引入 Muon 与 Lion)
    criterion = nn.MSELoss()
    muon_opt, lion_opt = get_muon_lion_optimizers(
        model,
        muon_lr=config["lr"],
        lion_lr=config["lion_lr"],
        weight_decay=1e-1,
    )
    scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(
        muon_opt, T_max=config["epochs"]
    )
    scheduler_lion = optim.lr_scheduler.CosineAnnealingLR(
        lion_opt, T_max=config["epochs"]
    )

    # 训练历史记录
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # 训练设置
    best_acc, best_loss = 0, float("inf")

    # 训练循环
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, muon_opt, lion_opt, scaler
        )
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 更新学习率
        scheduler_muon.step()
        scheduler_lion.step()

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, Path(config["save_dir"]) / "best_model_acc.pth")
            print("保存了新的最佳准确率模型!")

        # 保存最佳模型（基于损失）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, Path(config["save_dir"]) / "best_model_loss.pth")
            print("保存了新的最佳损失模型!")

        print(f"最佳准确率为: {best_acc:.2f}, 最佳损失为: {best_loss:.4f}")
        torch.save(model, Path(config["save_dir"]) / "best_model_full.pth")

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
        print(
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%"
        )

        # 计时
        if epoch == 0:
            start_time = time.time()
            epoch_start_time = start_time
        else:
            current_time = time.time()
            epoch_duration = current_time - epoch_start_time
            elapsed_time = current_time - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_time = (avg_epoch_time * config["epochs"]) - elapsed_time
            print(
                f"Epoch Time: {epoch_duration:.2f}s, Estimated Remaining: {remaining_time / 60:.2f}min"
            )
            epoch_start_time = current_time  # Reset for next epoch

        print("-" * 40)

    # 重命名与绘图
    current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_filename = f"data{data_length}_acc{best_acc:.4f}_loss{best_loss:.4f}_{current_time_str}.pth"
    save_dir_path = Path(config["save_dir"])

    for model_type in ["acc", "loss", "full"]:
        old_path = save_dir_path / f"best_model_{model_type}.pth"
        if old_path.exists():
            old_path.rename(
                save_dir_path / f"best_model_{model_type}_{base_filename}"
            )

    plot_learning_curve(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_dir_path / f"learning_curve_{base_filename}.png",
    )


if __name__ == "__main__":
    main()
