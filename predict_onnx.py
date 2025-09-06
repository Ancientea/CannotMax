import onnxruntime as ort
import os
import numpy as np
import logging

from config import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)

class CannotModel:
    def __init__(self,model_path = "models/best_model_full.onnx"):
        self.session = None  # ONNX Runtime 会话
        self.model_path = model_path
        try:
            self.load_model()  # 初始化时加载模型
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.session = None

    def load_model(self):
        """加载 ONNX 模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"未找到 ONNX 模型文件 {self.model_path}")
            
            # 配置会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 创建会话（默认使用 CPU）
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
        except Exception as e:
            raise RuntimeError(f"ONNX 模型加载失败: {str(e)}")

    def get_prediction(self, left_counts: np.ndarray, right_counts: np.ndarray):
        if self.session is None:
            raise RuntimeError("模型未正确初始化")
        
        def validate_input(arr):
            """验证并转换输入数据"""
            # 转换为 int64 类型
            arr = arr.astype(np.int64)
    
            # 添加批次维度（如果输入是单样本）
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # shape: (1, 56)
            return arr
        
        # 添加批次维度
        inputs = {
            "left_counts": validate_input(left_counts).astype(np.int64),
            "right_counts": validate_input(right_counts).astype(np.int64)
        }
        
        # 执行推理
        try:
            output = self.session.run(
                output_names=["output"],
                input_feed=inputs
            )
            print(output)
            prediction = output[0]
        except Exception as e:
            raise RuntimeError(f"推理失败: {str(e)}")
        
        # 后处理（与原逻辑一致）
        if np.isnan(prediction) or np.isinf(prediction):
            logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
            prediction = 0.5
        
        prediction = np.clip(prediction, 0.0, 1.0)
        return float(prediction)
    
    def get_prediction_with_terrain(self, full_features: np.ndarray):
        """使用包含地形特征的完整特征向量进行预测（ONNX版本）"""
        if self.session is None:
            raise RuntimeError("模型未正确初始化")

        # 检查特征向量长度
        expected_length = MONSTER_COUNT * 2 + FIELD_FEATURE_COUNT * 2  # 77L + 6L + 77R + 6R = 166
        if len(full_features) != expected_length:
            logger.warning(f"特征向量长度不匹配: 期望{expected_length}, 实际{len(full_features)}")
            # 如果长度不匹配，回退到原始方法
            left_counts = full_features[:MONSTER_COUNT]
            right_counts = full_features[MONSTER_COUNT:MONSTER_COUNT*2]
            return self.get_prediction(left_counts, right_counts)

        # 提取各个部分
        left_monsters = full_features[:MONSTER_COUNT]  # 1L-77L
        left_terrain = full_features[MONSTER_COUNT:MONSTER_COUNT+FIELD_FEATURE_COUNT]  # 78L-83L
        right_monsters = full_features[MONSTER_COUNT+FIELD_FEATURE_COUNT:MONSTER_COUNT*2+FIELD_FEATURE_COUNT]  # 1R-77R
        right_terrain = full_features[MONSTER_COUNT*2+FIELD_FEATURE_COUNT:MONSTER_COUNT*2+FIELD_FEATURE_COUNT*2]  # 78R-83R
        
        # 合并怪物特征和地形特征（按照训练时的格式）
        left_counts = np.concatenate([left_monsters, left_terrain])
        right_counts = np.concatenate([right_monsters, right_terrain])

        # 使用合并后的特征进行预测
        return self.get_prediction(left_counts, right_counts)