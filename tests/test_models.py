from __future__ import annotations

import pytest

from gradio_ml.infrastructure.models import OptimizerType, TrainingConfig, ValidationResult


class TestTrainingConfigValidate:
    def test_valid_config(self):
        config = TrainingConfig(learning_rate=0.001, epochs=100, batch_size=32)
        assert config.validate() == []

    def test_learning_rate_zero(self):
        config = TrainingConfig(learning_rate=0.0)
        errors = config.validate()
        assert any("学习率" in e for e in errors)

    def test_learning_rate_negative(self):
        config = TrainingConfig(learning_rate=-0.1)
        errors = config.validate()
        assert any("学习率" in e for e in errors)

    def test_learning_rate_above_one(self):
        config = TrainingConfig(learning_rate=1.5)
        errors = config.validate()
        assert any("学习率" in e for e in errors)

    def test_learning_rate_one_is_valid(self):
        config = TrainingConfig(learning_rate=1.0)
        assert all("学习率" not in e for e in config.validate())

    def test_epochs_zero(self):
        config = TrainingConfig(epochs=0)
        errors = config.validate()
        assert any("训练轮数" in e for e in errors)

    def test_epochs_above_max(self):
        config = TrainingConfig(epochs=10001)
        errors = config.validate()
        assert any("训练轮数" in e for e in errors)

    def test_batch_size_not_power_of_two(self):
        config = TrainingConfig(batch_size=3)
        errors = config.validate()
        assert any("批次大小" in e for e in errors)

    def test_batch_size_power_of_two(self):
        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            config = TrainingConfig(batch_size=bs)
            assert all("批次大小" not in e for e in config.validate())

    def test_batch_size_larger_than_dataset(self):
        config = TrainingConfig(batch_size=64, dataset_size=32)
        errors = config.validate()
        assert any("数据集大小" in e for e in errors)


class TestValidationResult:
    def test_success(self):
        result = ValidationResult.success()
        assert result.is_valid
        assert result.errors == []

    def test_failure(self):
        result = ValidationResult.failure(["error1", "error2"])
        assert not result.is_valid
        assert len(result.errors) == 2
