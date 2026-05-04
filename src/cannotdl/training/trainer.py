import time
from datetime import datetime
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cannotdl.config import FIELD_FEATURE_COUNT, MONSTER_COUNT
from cannotdl.models import (
    TOTAL_FEATURE_COUNT,
    ArknightsDataset,
    UnitAwareTransformer,
)
from cannotmax.config.paths import DATA_DIR, MODELS_DIR

from .muon import get_muon_lion_optimizers


@cache
def get_device(prefer_gpu=True):
    """
    prefer_gpu (bool): 是否优先尝试使用GPU
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    return torch.device("cpu")


device = get_device()


def plot_learning_curve(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制学习曲线并保存为图片"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss")
    plt.plot(epochs, val_losses, "r-", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

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
    print(f"预处理数据文件: {csv_file}")

    data = pd.read_csv(csv_file, header=None, skiprows=1)
    print(f"原始数据形状: {data.shape}")

    expected_columns = TOTAL_FEATURE_COUNT + 2
    if data.shape[1] != expected_columns:
        print(f"数据列数不符！期望 {expected_columns} 列，实际 {data.shape[1]} 列")
        print(
            f"期望格式: {MONSTER_COUNT}(怪物L) + {FIELD_FEATURE_COUNT}(场地L) + {MONSTER_COUNT}(怪物R) + {FIELD_FEATURE_COUNT}(场地R) + 1(Result) + 1(ImgPath)"
        )
        raise Exception("数据格式不符")

    data = data.iloc[:, 0 : TOTAL_FEATURE_COUNT + 1]

    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    extreme_values = (np.abs(features) > 20).sum().sum()
    if extreme_values > 0:
        print(f"发现 {extreme_values} 个绝对值大于20的特征值")

    invalid_labels = labels.apply(lambda x: x not in ["L", "R"]).sum()
    if invalid_labels > 0:
        print(f"发现 {invalid_labels} 个无效标签")

    feature_min = features.min().min()
    feature_max = features.max().max()
    feature_mean = features.mean().mean()
    feature_std = features.std().mean()

    print(f"特征值范围: [{feature_min}, {feature_max}]")
    print(f"特征值平均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")

    return data.shape[1]


def train_one_epoch(model, train_loader, criterion, muon_opt, lion_opt, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [
            x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)
        ]

        muon_opt.zero_grad()
        lion_opt.zero_grad()

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

        if (labels < 0).any() or (labels > 1).any():
            print("警告: 标签值不在[0,1]范围内，进行修正")
            labels = torch.clamp(labels, 0, 1)

        try:
            with torch.amp.autocast_mode.autocast(
                device_type=device.type, enabled=(scaler is not None)
            ):
                outputs = model(ls, lc, rs, rc).squeeze()
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("警告: 模型输出包含NaN或Inf，跳过该批次")
                    continue
                if (outputs < 0).any() or (outputs > 1).any():
                    print("警告: 模型输出不在[0,1]范围内，进行修正")
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

            loss = criterion(outputs.float(), labels.float())

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(muon_opt)
                scaler.unscale_(lion_opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(muon_opt)
                scaler.step(lion_opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                muon_opt.step()
                lion_opt.step()

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
            ls, lc, rs, rc, labels = [
                x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)
            ]

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

            if (labels < 0).any() or (labels > 1).any():
                labels = torch.clamp(labels, 0, 1)

            try:
                with torch.amp.autocast_mode.autocast(
                    device_type=device.type, enabled=(device.type == "cuda")
                ):
                    outputs = model(ls, lc, rs, rc).squeeze()
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print("警告: 评估时模型输出包含NaN或Inf，跳过该批次")
                        continue
                    if (outputs < 0).any() or (outputs > 1).any():
                        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs.float(), labels.float())

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
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
    )


def main():
    config = {
        "data_file": DATA_DIR / "arknights.csv",
        "batch_size": 1024,
        "test_size": 0.1,
        "embed_dim": 256,
        "n_layers": 3,
        "num_heads": 4,
        "dropout": 0.3,
        "lr": 3e-4,
        "lion_lr": 3e-4 / 10,
        "epochs": 50,
        "seed": 42,
        "save_dir": MODELS_DIR / "predictor",
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
        print("CUDA可用，已启用混合精度训练的GradScaler。")

    if str(device) == "cuda":
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif str(device) == "cpu":
        print("警告: 未检测到GPU，将在CPU上运行训练!")

    preprocess_data(config["data_file"])

    dataset = ArknightsDataset(
        config["data_file"],
        max_value=config["max_feature_value"],
    )

    data_length = len(dataset)
    val_size = int(0.1 * data_length)
    train_size = data_length - val_size

    train_dataset, val_dataset = stratified_random_split(
        dataset, test_size=config["test_size"], seed=config["seed"]
    )

    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    total_units = MONSTER_COUNT + FIELD_FEATURE_COUNT
    model = UnitAwareTransformer(
        num_units=total_units,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    print(
        f"模型使用特征数: 怪物({MONSTER_COUNT}) + 场地({FIELD_FEATURE_COUNT}) = {total_units}"
    )
    print(
        f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    criterion = nn.MSELoss()
    muon_opt, lion_opt = get_muon_lion_optimizers(
        model, muon_lr=config["lr"], lion_lr=config["lion_lr"], weight_decay=1e-1
    )
    scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(
        muon_opt, T_max=config["epochs"]
    )
    scheduler_lion = optim.lr_scheduler.CosineAnnealingLR(
        lion_opt, T_max=config["epochs"]
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_acc, best_loss = 0, float("inf")

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, muon_opt, lion_opt, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler_muon.step()
        scheduler_lion.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, Path(config["save_dir"]) / "best_model_acc.pth")
            print("保存了新的最佳准确率模型!")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, Path(config["save_dir"]) / "best_model_loss.pth")
            print("保存了新的最佳损失模型!")

        print(f"最佳准确率为: {best_acc:.2f}, 最佳损失为: {best_loss:.4f}")
        torch.save(model, Path(config["save_dir"]) / "best_model_full.pth")

        print(
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%    Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%"
        )

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
            epoch_start_time = current_time

        print("-" * 40)

    current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_filename = f"data{data_length}_acc{best_acc:.4f}_loss{best_loss:.4f}_{current_time_str}.pth"
    save_dir_path = Path(config["save_dir"])

    for model_type in ["acc", "loss", "full"]:
        old_path = save_dir_path / f"best_model_{model_type}.pth"
        if old_path.exists():
            old_path.rename(save_dir_path / f"best_model_{model_type}_{base_filename}")

    plot_learning_curve(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_dir_path / f"learning_curve_{base_filename}.png",
    )


if __name__ == "__main__":
    main()

# test lint hook
