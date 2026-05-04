from pathlib import Path

import onnxruntime as ort
import os
import numpy as np
import logging

from config import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT

logger = logging.getLogger(__name__)

class CannotModel:
    def __init__(self, model_path="models"):
        self.is_model_loaded = False
        self.sessions = []
        self.model_path = model_path
        
        # 自动扫描 models/ 下所有 .onnx 文件
        self.sessions = []
        onnx_files = sorted(Path(model_path).glob("*.onnx"))
        for p in onnx_files:
            try:
                sess = self._create_session(str(p))
                self.sessions.append(sess)
                logger.info(f"加载 ONNX: {p.name}")
            except Exception as e:
                logger.warning(f"跳过 ONNX {p.name}: {e}")
            
            if self.sessions:
                self.is_model_loaded = True
                logger.info(f"集成模型加载完成: {len(self.sessions)} 个")
            else:
                logger.error("未找到任何 ONNX 模型文件")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.sessions = []

    def _create_session(self, path):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, sess_options, providers=['CPUExecutionProvider'])

    def _resolve_model_path(self, path):
        return str(Path(path) / "best_model_full.onnx")

    def get_prediction(self, left_counts: np.ndarray, right_counts: np.ndarray):
        if not self.sessions:
            raise RuntimeError("模型未正确初始化")
        
        def validate_input(arr):
            arr = arr.astype(np.int64)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return arr
        
        left_signs_arr = np.sign(left_counts).astype(np.int64)
        left_counts_arr = np.abs(left_counts).astype(np.int64)
        right_signs_arr = np.sign(right_counts).astype(np.int64)
        right_counts_arr = np.abs(right_counts).astype(np.int64)

        inputs = {
            "left_signs": validate_input(left_signs_arr),
            "left_counts": validate_input(left_counts_arr),
            "right_signs": validate_input(right_signs_arr),
            "right_counts": validate_input(right_counts_arr)
        }
        
        predictions = []
        for sess in self.sessions:
            output = sess.run(output_names=["output"], input_feed=inputs)
            p = output[0].flatten()[0]
            if not (np.isnan(p) or np.isinf(p)):
                predictions.append(p)
        
        if not predictions:
            return 0.5
        prediction = float(np.mean(predictions))
        return float(np.clip(prediction, 0.0, 1.0))
    
    def get_prediction_with_terrain(self, full_features: np.ndarray):
        """使用包含地形特征的完整特征向量进行预测（ONNX版本）"""
        if not self.sessions:
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

        # 处理左侧特征
        left_monster_signs = np.sign(left_monsters).astype(np.int64)
        left_terrain_signs = np.ones_like(left_terrain).astype(np.int64)
        left_signs = np.concatenate([left_monster_signs, left_terrain_signs])

        left_monster_counts = np.abs(left_monsters).astype(np.int64)
        left_counts = np.concatenate([left_monster_counts, left_terrain.astype(np.int64)])

        # 处理右侧特征
        right_monster_signs = np.sign(right_monsters).astype(np.int64)
        right_terrain_signs = np.ones_like(right_terrain).astype(np.int64)
        right_signs = np.concatenate([right_monster_signs, right_terrain_signs])

        right_monster_counts = np.abs(right_monsters).astype(np.int64)
        right_counts = np.concatenate([right_monster_counts, right_terrain.astype(np.int64)])

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
            "right_counts": validate_input(right_counts)
        }

        predictions = []
        for sess in self.sessions:
            output = sess.run(output_names=["output"], input_feed=inputs)
            p = output[0].flatten()[0]
            if not (np.isnan(p) or np.isinf(p)):
                predictions.append(p)
        
        if not predictions:
            logger.warning("所有模型输出异常，返回默认值0.5")
            return 0.5
        prediction = float(np.mean(predictions))
        prediction = np.clip(prediction, 0.0, 1.0)
        return float(prediction)