"""Unit tests for CannotModel prediction."""

import pytest
import numpy as np
from pathlib import Path
from src.cannotmax.core.predict import CannotModel
from src.cannotmax.config import MONSTER_COUNT
from src.cannotmax.config.paths import MODELS_DIR


@pytest.fixture(scope="module")
def model():
    if not list((MODELS_DIR / "predictor").glob("*.pth")):
        pytest.skip("No model checkpoint found in models/")
    return CannotModel()


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
