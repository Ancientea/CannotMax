import os
import numpy as np
import torch

from recognize import MONSTER_COUNT


class CannotModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # 模型实例
        self.load_model()  # 初始化时加载模型
        pass

    def load_model(self):
        """初始化时加载模型"""
        try:
            if not os.path.exists("models/best_model_full.pth"):
                raise FileNotFoundError(
                    "未找到训练好的模型文件 'models/best_model_full.pth'，请先训练模型"
                )

            try:
                model = torch.load(
                    "models/best_model_full.pth", map_location=self.device, weights_only=False
                )
            except TypeError:  # 如果旧版本 PyTorch 不认识 weights_only
                model = torch.load("models/best_model_full.pth", map_location=self.device)
            model.eval()
            self.model = model.to(self.device)

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            if "missing keys" in str(e):
                error_msg += "\n可能是模型结构不匹配，请重新训练模型"
            raise e  # 无法继续运行，退出程序

    def get_prediction(self, left_monsters, right_monsters):
        if self.model is None:
            raise RuntimeError("模型未正确初始化")

        # 准备输入数据（完全匹配ArknightsDataset的处理方式）
        left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

        # 从界面获取数据（空值处理为0）
        for name, entry in left_monsters.items():
            value = entry.get()
            left_counts[int(name) - 1] = int(value) if value.isdigit() else 0

        for name, entry in right_monsters.items():
            value = entry.get()
            right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

        # 转换为张量并处理符号和绝对值
        left_signs = (
            torch.sign(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        )
        left_counts = (
            torch.abs(torch.tensor(left_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        )
        right_signs = (
            torch.sign(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        )
        right_counts = (
            torch.abs(torch.tensor(right_counts, dtype=torch.int16)).unsqueeze(0).to(self.device)
        )

        # 预测流程
        with torch.no_grad():
            # 使用修改后的模型前向传播流程
            prediction = self.model(left_signs, left_counts, right_signs, right_counts).item()

            # 确保预测值在有效范围内
            if np.isnan(prediction) or np.isinf(prediction):
                print("警告: 预测结果包含NaN或Inf，返回默认值0.5")
                prediction = 0.5

            # 检查预测结果是否在[0,1]范围内
            if prediction < 0 or prediction > 1:
                prediction = max(0, min(1, prediction))

        return prediction
