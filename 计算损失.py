"""
模型 BCELoss 评估脚本
用法:
  python eval_bce.py                                    # 默认 filtered 数据
  python eval_bce.py arknights_cleaned.csv              # 用清洗后数据
  python eval_bce.py train.csv val.csv                  # 独立训练集+验证集(分测)
"""
import sys
from pathlib import Path
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import UnitAwareTransformer, ArknightsDataset, get_device
import train as _train

DATA_FILE = "arknights_filtered.csv"
VAL_FILE = None
MODEL_DIR = Path("models")
BATCH_SIZE = 4096

# 兼容旧 pickle 引用
import importlib, train as _train
if not hasattr(_train, "EnsembleModel"):
    _train.EnsembleModel = UnitAwareTransformer
sys.modules["models.model"] = _train
sys.modules.setdefault("ensemble_model", _train)
try:
    importlib.import_module("models")
except ImportError:
    pass


def progress_bar(current, total, label="", width=30):
    """简单的终端进度条，用 \r 原地刷新"""
    pct = current / total if total else 1
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    # 文件名截断，保证不换行
    label = label if len(label) <= 36 else "..." + label[-33:]
    print(f"\r[{bar}] {current}/{total} {label:<36}", end="", flush=True)


def load_model(path, device):
    m = torch.load(str(path), map_location=device, weights_only=False)
    # 检测 state_dict（仅含权重，非完整模型）
    if isinstance(m, dict):
        # 尝试按 UnitAwareTransformer 键名判断
        if any('unit_embed' in k for k in m.keys()):
            # UnitAwareTransformer 的 state_dict → 重建模型
            model = UnitAwareTransformer()
            model.load_state_dict(m, strict=False)
            model.eval().to(device)
            return model
        elif any('image_encoder' in k for k in m.keys()):
            raise RuntimeError("旧版图像模型(ViT)，非当前架构")
        else:
            raise RuntimeError("未知 state_dict 格式")
    if not hasattr(m, 'unit_embed'):
        raise RuntimeError("旧架构(无 unit_embed)")
    m.eval().to(device)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    criterion = nn.BCELoss()
    total_loss, correct, total = 0.0, 0, 0
    for ls, lc, rs, rc, labels in loader:
        ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]
        outputs = model(ls, lc, rs, rc).squeeze()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    device = get_device()
    print(f"设备: {device}")

    args = sys.argv[1:]

    if len(args) >= 2:
        data_file = args[0]
        val_file = args[1]
        print(f"训练集: {data_file}")
        print(f"验证集: {val_file} (独立)")

        ds_train = ArknightsDataset(data_file, max_value=100)
        ds_val   = ArknightsDataset(val_file, max_value=100)
        loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"训练集样本: {len(ds_train)}, 验证集样本: {len(ds_val)}")
    else:
        data_file = args[0] if args else DATA_FILE
        print(f"验证集: {data_file}")
        ds = ArknightsDataset(data_file, max_value=100)
        loader_val = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        loader_train = None
        print(f"样本: {len(ds)} (纯验证，不拆分)")

    # ── 加载模型 ──
    model_files = sorted(MODEL_DIR.glob("*.pth"))
    if not model_files:
        print(f"\n{MODEL_DIR} 下无 .pth 文件")
        return

    # ── 先收集所有结果 ──
    results = []  # (name, loss, acc)
    total = len(model_files)
    print(f"\n共 {total} 个模型待评估\n")
    for i, f in enumerate(model_files, 1):
        progress_bar(i, total, f.name)
        try:
            model = load_model(f, device)
            vl_loss, vl_acc = evaluate(model, loader_val, device)
            results.append((f.name, vl_loss, vl_acc))
        except Exception as e:
            print(f"\n{f.name:<45} 跳过: {e}")
    print()  # 换行

    if not results:
        print("\n没有成功评估的模型")
        return

    # ── 按损失升序排列 ──
    results.sort(key=lambda x: x[1])

    print(f"\n{'模型':<45} {'验证Loss':>10} {'验证Acc':>9}")
    print("-" * 67)
    for name, loss, acc in results:
        print(f"{name:<45} {loss:>10.6f} {acc:>8.2f}%")

    best_loss = results[0]
    best_acc = max(results, key=lambda x: x[2])
    print(f"\n最低Loss: {best_loss[0]} ({best_loss[1]:.6f})")
    print(f"最高Acc:  {best_acc[0]} ({best_acc[2]:.2f}%)")


if __name__ == "__main__":
    main()
