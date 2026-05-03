"""Unit tests for CannotModel prediction."""

import numpy as np
import pytest

from cannotmax.config import MONSTER_COUNT
from cannotmax.config.paths import MODELS_DIR
from cannotmax.core import CannotModel


@pytest.fixture(scope="module")
def model():
    if not list((MODELS_DIR / "predictor").glob("*.onnx")):
        pytest.skip("No ONNX model found in models/predictor/")
    m = CannotModel(model_path=MODELS_DIR / "predictor")
    if not m.is_model_loaded:
        pytest.skip("ONNX model failed to load (no valid checkpoint)")
    return m


class TestCannotModel:
    def test_model_loads(self, model):
        assert model.is_model_loaded
        assert model.model is not None

    def test_prediction_returns_float(self, model):
        left = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right = np.zeros(MONSTER_COUNT, dtype=np.int16)
        left[0] = 5
        result = model.get_prediction(left, right)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_prediction_symmetric_inputs(self, model):
        counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        counts[0] = 10
        counts[1] = 20
        r1 = model.get_prediction(counts, np.zeros_like(counts))
        r2 = model.get_prediction(np.zeros_like(counts), counts)
        assert isinstance(r1, float)
        assert isinstance(r2, float)
