import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import UnitAwareTransformer,ArknightsDataset

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            # 确保数据在CPU上，然后移动到设备
            ls, lc, rs, rc, labels = [x.to(device) for x in batch]

            # 其余代码保持不变...
            # 检查输入值范围
            if torch.isnan(ls).any() or torch.isnan(lc).any() or torch.isnan(rs).any() or torch.isnan(rc).any() or \
                    torch.isinf(ls).any() or torch.isinf(lc).any() or torch.isinf(rs).any() or torch.isinf(rc).any():
                print("警告: 评估时输入数据包含NaN或Inf，跳过该批次")
                continue

            # 确保labels严格在0-1之间
            if (labels < 0).any() or (labels > 1).any():
                labels = torch.clamp(labels, 0, 1)

            try:
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

def main():
    config = {
        'data_file': 'arknights_clean.csv',
        'batch_size': 256,
        'max_feature_value': 200  # 限制特征最大值，防止极端值造成不稳定
    }
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

        # 设置确定性计算以增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("警告: 未检测到GPU，将在CPU上运行训练，这可能会很慢!")
        
    dataset = ArknightsDataset(
        config['data_file'],
        max_value=config['max_feature_value']  # 使用最大值限制
    )
    
    # 修改这里 - 只有当数据在CPU上时才使用pin_memory
    val_loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=False  # 禁用pin_memory，或者根据你的数据集情况调整
    )
    
    model = UnitAwareTransformer(
        num_units=34,  # 更新为34个怪物
        embed_dim=128,
        num_heads=8,
        num_layers=4  # 注意：train.py中config['n_layers']=4
    ).to(device)

    # 加载模型权重
    try:
        model = torch.load('models/best_model_full.pth', map_location=device, weights_only=False)
    except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
        model = torch.load('models/best_model_full.pth', map_location=device)
    model.eval()
    criterion = nn.BCELoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
    
if __name__ == "__main__":
    main()