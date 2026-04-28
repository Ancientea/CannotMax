from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gradio_ml.infrastructure.models import (
    MetricDataPoint,
    MetricQuery,
    PredictInput,
    PredictResult,
    TrainingPhase,
    ValidationResult,
)
from gradio_ml.service.predict_service import PredictService


class TestPredictServiceValidate:
    def test_valid_left_right(self):
        svc = PredictService.__new__(PredictService)
        svc._timeout = 60
        svc._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        inp = PredictInput(left_counts=[1, 2, 3], right_counts=[4, 5, 6])
        result = svc.validate_input(inp)
        assert result.is_valid

    def test_valid_full_features(self):
        svc = PredictService.__new__(PredictService)
        svc._timeout = 60
        svc._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        inp = PredictInput(full_features=[1.0] * 166)
        result = svc.validate_input(inp)
        assert result.is_valid

    def test_empty_left_counts(self):
        svc = PredictService.__new__(PredictService)
        svc._timeout = 60
        svc._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        inp = PredictInput(left_counts=[], right_counts=[1, 2])
        result = svc.validate_input(inp)
        assert not result.is_valid

    def test_empty_right_counts(self):
        svc = PredictService.__new__(PredictService)
        svc._timeout = 60
        svc._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        inp = PredictInput(left_counts=[1, 2], right_counts=[])
        result = svc.validate_input(inp)
        assert not result.is_valid

    def test_empty_full_features(self):
        svc = PredictService.__new__(PredictService)
        svc._timeout = 60
        svc._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        inp = PredictInput(full_features=[])
        result = svc.validate_input(inp)
        assert not result.is_valid


class TestPredictServiceListModels:
    def test_list_models_nonexistent_dir(self):
        svc = PredictService.__new__(PredictService)
        svc._adapter = MagicMock()
        result = svc.list_available_models("/nonexistent/path")
        assert result == []
