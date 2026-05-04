"""
将训练好的 .pth 模型导出为 ONNX 格式，供 GUI 使用。
用法: python export_onnx.py [模型路径]
默认加载 models/model_seed42_swa.pth
"""
import sys
import torch
from pathlib import Path

# 从 train.py 导入模型类
from train import UnitAwareTransformer, get_device, MONSTER_COUNT, FIELD_FEATURE_COUNT

device = get_device()

def export_to_onnx(pth_path, onnx_path):
    print(f"加载模型: {pth_path}")
    
    # 创建模型结构
    total_units = MONSTER_COUNT + FIELD_FEATURE_COUNT
    model = UnitAwareTransformer(
        num_units=total_units,
        embed_dim=128,
        num_heads=16,
        num_layers=3,
        dropout=0.0,  # 推理时不需要 dropout
    )
    
    # 加载权重
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint  # 完整模型保存
    
    model.to(device)
    model.eval()
    
    # 创建虚拟输入
    N = MONSTER_COUNT + FIELD_FEATURE_COUNT
    dummy_left_signs = torch.ones(1, N, device=device)
    dummy_left_counts = torch.ones(1, N, device=device)
    dummy_right_signs = torch.ones(1, N, device=device)
    dummy_right_counts = torch.ones(1, N, device=device)
    
    print(f"导出 ONNX 到: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_left_signs, dummy_left_counts, dummy_right_signs, dummy_right_counts),
        onnx_path,
        input_names=["left_signs", "left_counts", "right_signs", "right_counts"],
        output_names=["output"],
        dynamic_axes={
            "left_signs": {0: "batch"},
            "left_counts": {0: "batch"},
            "right_signs": {0: "batch"},
            "right_counts": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=14,
    )
    print("导出成功！")

if __name__ == "__main__":
    pth_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_seed42_swa.pth"
    onnx_path = str(Path(pth_path).with_suffix(".onnx"))
    export_to_onnx(pth_path, onnx_path)
