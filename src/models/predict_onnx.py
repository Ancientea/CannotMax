from pathlib import Path

import onnxruntime as ort
import os
import numpy as np
import logging

from src.core.config import MONSTER_COUNT
from src.core.config import FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)


class CannotModel:
    def __init__(self, model_path="models"):
        self.model_path = self._resolve_model_path(model_path)
        self.is_model_loaded = False
        try:
            self.load_model()  # 初始化时加载模型
            self.is_model_loaded = True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.session = None

    def _resolve_model_path(self, path):
        """
        Resolves the model path. If a directory is given, finds the latest model file.
        If a file is given, returns it directly.
        """
        if Path(path).is_dir():
            logger.info(f"Searching for the latest model in directory: {path}")
            model_dir = Path(path)

            # 尝试寻找默认的 best_model_full.onnx
            default_path = model_dir / "best_model_full.onnx"
            if default_path.exists():
                logger.info(f"Found default model: {default_path}")
                return str(default_path)

            logger.error(f"No valid ONNX model files found in {path}")
            return str(default_path)

        elif Path(path).is_file():
            logger.info(f"Using specified model file: {path}")
            return path
        else:
            logger.error(f"Provided model path is invalid: {path}")
            return ""

    def load_model(self):
        """加载 ONNX 模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"未找到 ONNX 模型文件 {self.model_path}"
                )

            # 配置会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # 创建会话（默认使用 CPU）
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )

        except Exception as e:
            raise RuntimeError(f"ONNX 模型加载失败: {str(e)}")

    def get_prediction(
        self, left_counts: np.ndarray, right_counts: np.ndarray
    ):
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

        # 处理符号和绝对值，以匹配导出的模型输入
        left_signs_arr = np.sign(left_counts).astype(np.int64)
        left_counts_arr = np.abs(left_counts).astype(np.int64)
        right_signs_arr = np.sign(right_counts).astype(np.int64)
        right_counts_arr = np.abs(right_counts).astype(np.int64)

        inputs = {
            "left_signs": validate_input(left_signs_arr),
            "left_counts": validate_input(left_counts_arr),
            "right_signs": validate_input(right_signs_arr),
            "right_counts": validate_input(right_counts_arr),
        }

        # 执行推理
        try:
            output = self.session.run(
                output_names=["output"], input_feed=inputs
            )
            # output 是一个列表，output[0] 是形状为 (batch_size, 1) 的数组
            prediction = output[0].flatten()[0]
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
        expected_length = (
            MONSTER_COUNT * 2 + FIELD_FEATURE_COUNT * 2
        )  # 77L + 6L + 77R + 6R = 166
        if len(full_features) != expected_length:
            logger.warning(
                f"特征向量长度不匹配: 期望{expected_length}, 实际{len(full_features)}"
            )
            # 如果长度不匹配，回退到原始方法
            left_counts = full_features[:MONSTER_COUNT]
            right_counts = full_features[MONSTER_COUNT : MONSTER_COUNT * 2]
            return self.get_prediction(left_counts, right_counts)

        # 提取各个部分
        left_monsters = full_features[:MONSTER_COUNT]  # 1L-77L
        left_terrain = full_features[
            MONSTER_COUNT : MONSTER_COUNT + FIELD_FEATURE_COUNT
        ]  # 78L-83L
        right_monsters = full_features[
            MONSTER_COUNT
            + FIELD_FEATURE_COUNT : MONSTER_COUNT * 2
            + FIELD_FEATURE_COUNT
        ]  # 1R-77R
        right_terrain = full_features[
            MONSTER_COUNT * 2
            + FIELD_FEATURE_COUNT : MONSTER_COUNT * 2
            + FIELD_FEATURE_COUNT * 2
        ]  # 78R-83R

        # 处理左侧特征
        left_monster_signs = np.sign(left_monsters).astype(np.int64)
        left_terrain_signs = np.ones_like(left_terrain).astype(np.int64)
        left_signs = np.concatenate([left_monster_signs, left_terrain_signs])

        left_monster_counts = np.abs(left_monsters).astype(np.int64)
        left_counts = np.concatenate(
            [left_monster_counts, left_terrain.astype(np.int64)]
        )

        # 处理右侧特征
        right_monster_signs = np.sign(right_monsters).astype(np.int64)
        right_terrain_signs = np.ones_like(right_terrain).astype(np.int64)
        right_signs = np.concatenate(
            [right_monster_signs, right_terrain_signs]
        )

        right_monster_counts = np.abs(right_monsters).astype(np.int64)
        right_counts = np.concatenate(
            [right_monster_counts, right_terrain.astype(np.int64)]
        )

        def validate_input(arr):
            """验证并转换输入数据"""
            arr = arr.astype(np.int64)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return arr

        inputs = {
            "left_signs": validate_input(left_signs),
            "left_counts": validate_input(left_counts),
            "right_signs": validate_input(right_signs),
            "right_counts": validate_input(right_counts),
        }

        # 执行推理
        try:
            output = self.session.run(
                output_names=["output"], input_feed=inputs
            )
            # output[0] 是形状为 (batch_size, 1) 的数组
            prediction = output[0].flatten()[0]
        except Exception as e:
            raise RuntimeError(f"推理失败: {str(e)}")

        # 后处理（与原逻辑一致）
        if np.isnan(prediction) or np.isinf(prediction):
            logger.warning("警告: 预测结果包含NaN或Inf，返回默认值0.5")
            prediction = 0.5

        prediction = np.clip(prediction, 0.0, 1.0)
        return float(prediction)
