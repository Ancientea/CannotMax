import numpy as np

from cannotdl.config import MONSTER_COUNT
from cannotdl.core import predict as torch_predict
from cannotmax.config.paths import DEFAULT_PREDICTOR_PTH
from cannotmax.core import predict_onnx

model_path = DEFAULT_PREDICTOR_PTH


def replace_suffix(path):
    return path.with_suffix(".onnx")


output_path = replace_suffix(model_path)

model = torch_predict.CannotModel(model_path)
model.load_model()  # 加载原始 PyTorch 模型
model.export_onnx(output_path)  # 导出 ONNX 模型
print(f"模型已成功导出为 ONNX 格式，保存路径: {output_path}")

# 验证导出结果
data_array = np.zeros(MONSTER_COUNT * 2, dtype=np.int64)
data_array[28] = 16
data_array[MONSTER_COUNT + 30] = 22
left = data_array[:MONSTER_COUNT]
right = data_array[MONSTER_COUNT:]

Onnxmodel = predict_onnx.CannotModel(model_path=output_path)
prediction = Onnxmodel.get_prediction(left, right)
print("预测结果:", prediction)
