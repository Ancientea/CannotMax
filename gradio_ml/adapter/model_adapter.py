from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gradio_ml.infrastructure.models import FrameworkType, ModelInfo, ModelSignature, ModelStatus

logger = logging.getLogger(__name__)


class ModelAdapter:
    def __init__(self):
        self._model = None
        self._model_info: ModelInfo | None = None

    def load(self, model_path: str) -> ModelInfo:
        from predict import CannotModel

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        try:
            model = CannotModel(model_path=model_path)
            self._model = model
            model_name = Path(model.model_path).name if model.model_path else str(model_path)
            self._model_info = ModelInfo(
                name=model_name,
                path=model.model_path or model_path,
                status=ModelStatus.Ready if model.is_model_loaded else ModelStatus.Error,
                signature=ModelSignature(
                    input_names=["left_signs", "left_counts", "right_signs", "right_counts"],
                    output_names=["output"],
                ),
                framework=FrameworkType.PyTorch,
            )
            logger.info(f"模型加载成功: {model_name}")
            return self._model_info
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model_info = ModelInfo(
                name=str(model_path),
                path=str(model_path),
                status=ModelStatus.Error,
                framework=FrameworkType.Unknown,
            )
            raise

    def infer(self, left_counts: np.ndarray, right_counts: np.ndarray) -> float:
        if self._model is None or not self._model.is_model_loaded:
            raise RuntimeError("模型未加载，无法执行推理")
        try:
            result = self._model.get_prediction(left_counts, right_counts)
            return float(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.warning("GPU OOM，回退至CPU推理")
                try:
                    self._model.device = __import__("torch").device("cpu")
                    self._model.model = self._model.model.cpu()
                    result = self._model.get_prediction(left_counts, right_counts)
                    return float(result)
                except Exception as fallback_err:
                    logger.error(f"CPU回退推理也失败: {fallback_err}")
                    raise
            raise

    def infer_with_terrain(self, full_features: np.ndarray) -> float:
        if self._model is None or not self._model.is_model_loaded:
            raise RuntimeError("模型未加载，无法执行推理")
        try:
            result = self._model.get_prediction_with_terrain(full_features)
            return float(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.warning("GPU OOM，回退至CPU推理")
                try:
                    self._model.device = __import__("torch").device("cpu")
                    self._model.model = self._model.model.cpu()
                    result = self._model.get_prediction_with_terrain(full_features)
                    return float(result)
                except Exception as fallback_err:
                    logger.error(f"CPU回退推理也失败: {fallback_err}")
                    raise
            raise

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._model.is_model_loaded

    @property
    def model_info(self) -> ModelInfo | None:
        return self._model_info

    def get_signature(self) -> ModelSignature | None:
        if self._model_info is None:
            return None
        return self._model_info.signature
