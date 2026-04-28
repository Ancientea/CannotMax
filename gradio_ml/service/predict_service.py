from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path

import numpy as np

from gradio_ml.adapter.model_adapter import ModelAdapter
from gradio_ml.infrastructure.models import (
    ModelInfo,
    PredictInput,
    PredictResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class PredictService:
    def __init__(self, model_adapter: ModelAdapter | None = None, inference_timeout: int = 60):
        self._adapter = model_adapter or ModelAdapter()
        self._timeout = inference_timeout
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def load_model(self, model_path: str) -> ModelInfo:
        return self._adapter.load(model_path)

    def predict(self, input_data: PredictInput) -> PredictResult:
        validation = self.validate_input(input_data)
        if not validation.is_valid:
            raise ValueError(f"输入校验失败: {', '.join(validation.errors)}")

        if not self._adapter.is_loaded:
            raise RuntimeError("模型未加载")

        model_name = self._adapter.model_info.name if self._adapter.model_info else "unknown"
        start_time = time.monotonic()

        try:
            future = self._executor.submit(self._do_inference, input_data)
            prediction = future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"推理超时（{self._timeout}秒）")

        elapsed_ms = (time.monotonic() - start_time) * 1000
        label = "胜" if prediction > 0.5 else "负"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return PredictResult(
            prediction=prediction,
            label=label,
            confidence=confidence,
            inference_time_ms=elapsed_ms,
            model_name=model_name,
        )

    def batch_predict(self, input_data_list: list[PredictInput]) -> list[PredictResult]:
        return [self.predict(inp) for inp in input_data_list]

    def list_available_models(self, model_dir: str = "models") -> list[str]:
        path = Path(model_dir)
        if not path.exists():
            return []
        return [str(f) for f in path.rglob("*.pth") if f.is_file()]

    def validate_input(self, input_data: PredictInput) -> ValidationResult:
        errors: list[str] = []
        if input_data.full_features is not None:
            if len(input_data.full_features) == 0:
                errors.append("完整特征向量不能为空")
        else:
            if not input_data.left_counts:
                errors.append("左侧数据不能为空")
            if not input_data.right_counts:
                errors.append("右侧数据不能为空")
        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()

    def _do_inference(self, input_data: PredictInput) -> float:
        if input_data.full_features is not None:
            features = np.array(input_data.full_features, dtype=np.float64)
            return self._adapter.infer_with_terrain(features)
        left = np.array(input_data.left_counts, dtype=np.float64)
        right = np.array(input_data.right_counts, dtype=np.float64)
        return self._adapter.infer(left, right)
