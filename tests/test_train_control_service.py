from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gradio_ml.infrastructure.models import OptimizerType, TrainingConfig, TrainingSessionStatus, ValidationResult
from gradio_ml.service.train_control_service import TrainControlService


class TestTrainControlServiceValidate:
    def test_valid_config(self):
        svc = TrainControlService()
        config = TrainingConfig(learning_rate=0.001, epochs=100, batch_size=32)
        result = svc.validate_config(config)
        assert result.is_valid

    def test_invalid_config(self):
        svc = TrainControlService()
        config = TrainingConfig(learning_rate=0.0, epochs=0, batch_size=3)
        result = svc.validate_config(config)
        assert not result.is_valid
        assert len(result.errors) > 1


def _long_training_fn(config, callback, pause_evt, stop_evt):
    import time
    for i in range(1000):
        if stop_evt.is_set():
            return
        while pause_evt.is_set():
            if stop_evt.is_set():
                return
            time.sleep(0.05)
        time.sleep(0.1)


class TestTrainControlServiceLifecycle:
    def test_start_training(self):
        svc = TrainControlService()
        config = TrainingConfig(learning_rate=0.001, epochs=2, batch_size=4)
        session = svc.start_training(config, training_fn=_long_training_fn)
        assert session.status == TrainingSessionStatus.Running
        import time
        time.sleep(0.3)
        svc.stop_training(session.session_id)

    def test_start_with_invalid_config_raises(self):
        svc = TrainControlService()
        config = TrainingConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="参数校验失败"):
            svc.start_training(config)

    def test_pause_and_resume(self):
        svc = TrainControlService()
        config = TrainingConfig(learning_rate=0.001, epochs=2, batch_size=4)
        session = svc.start_training(config, training_fn=_long_training_fn)
        import time
        time.sleep(0.2)
        paused = svc.pause_training(session.session_id)
        assert paused.status == TrainingSessionStatus.Paused
        resumed = svc.resume_training(session.session_id)
        assert resumed.status == TrainingSessionStatus.Running
        time.sleep(0.2)
        svc.stop_training(session.session_id)

    def test_stop_training(self):
        svc = TrainControlService()
        config = TrainingConfig(learning_rate=0.001, epochs=2, batch_size=4)
        session = svc.start_training(config, training_fn=_long_training_fn)
        import time
        time.sleep(0.2)
        stopped = svc.stop_training(session.session_id, save_model=False)
        assert stopped.status == TrainingSessionStatus.Terminated
