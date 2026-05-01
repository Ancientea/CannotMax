import sys
sys.path.append(".")

import predict
from train import UnitAwareTransformer
import predict_onnx
import numpy as np
from recognize import MONSTER_COUNT

model_path = "models/best_model_full.pth"

def replace_suffix(s):
    idx = s.rfind('.')
    if idx == -1:
        return s + '.onnx'
    else:
        return s[:idx] + '.onnx'

output_path = replace_suffix(model_path)

model = predict.CannotModel(model_path)
model.load_model()  # 加载原始 PyTorch 模型
model.export_onnx(output_path)  # 导出 ONNX 模型
print(f"模型已成功导出为 ONNX 格式，保存路径: {output_path}")

#验证导出结果
data_array = np.zeros(MONSTER_COUNT * 2, dtype=np.int64)
data_array[28] = 16
data_array[MONSTER_COUNT + 30] = 22
left = data_array[:MONSTER_COUNT]
right = data_array[MONSTER_COUNT:]

Onnxmodel = predict_onnx.CannotModel(model_path = output_path)
prediction = Onnxmodel.get_prediction(left, right)
print("预测结果:", prediction)
