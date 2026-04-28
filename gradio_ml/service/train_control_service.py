from __future__ import annotations

import logging
import threading
from uuid import UUID

from gradio_ml.adapter.ml_framework_adapter import MLFrameworkAdapter
from gradio_ml.infrastructure.models import (
    OptimizerType,
    TrainingConfig,
    TrainingSession,
    TrainingSessionStatus,
    ValidationResult,
)
from gradio_ml.infrastructure.queues import clear_queues, pause_event, reset_events, stop_event
from gradio_ml.infrastructure.session_manager import TrainingSessionManager

logger = logging.getLogger(__name__)


class TrainControlService:
    def __init__(self, session_manager: TrainingSessionManager | None = None):
        self._session_mgr = session_manager or TrainingSessionManager()
        self._training_thread: threading.Thread | None = None

    def validate_config(self, config: TrainingConfig) -> ValidationResult:
        errors = config.validate()
        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()

    def start_training(
        self,
        config: TrainingConfig,
        training_fn=None,
    ) -> TrainingSession:
        validation = self.validate_config(config)
        if not validation.is_valid:
            raise ValueError(f"参数校验失败: {', '.join(validation.errors)}")

        active = self._session_mgr.get_active_session()
        if active is not None and active.status == TrainingSessionStatus.Running:
            raise RuntimeError("已有训练正在运行")

        session = self._session_mgr.create_session(config)
        self._session_mgr.transition(session.session_id, TrainingSessionStatus.Running)

        reset_events()
        clear_queues()
        pause_event.clear()
        stop_event.clear()

        framework = MLFrameworkAdapter.detect_framework()
        callback = MLFrameworkAdapter.create_callback(session.session_id, framework)

        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(session, config, training_fn, callback),
            daemon=True,
        )
        self._training_thread.start()
        logger.info(f"训练已启动: 会话 {session.session_id}")
        return session

    def pause_training(self, session_id: UUID) -> TrainingSession:
        session = self._session_mgr.transition(session_id, TrainingSessionStatus.Paused)
        pause_event.set()
        logger.info(f"训练已暂停: 会话 {session_id}")
        return session

    def resume_training(self, session_id: UUID) -> TrainingSession:
        session = self._session_mgr.transition(session_id, TrainingSessionStatus.Running)
        pause_event.clear()
        logger.info(f"训练已恢复: 会话 {session_id}")
        return session

    def stop_training(self, session_id: UUID, save_model: bool = False) -> TrainingSession:
        stop_event.set()
        pause_event.clear()
        session = self._session_mgr.transition(session_id, TrainingSessionStatus.Terminated)
        if save_model:
            try:
                logger.info("正在保存模型...")
            except Exception as e:
                logger.error(f"模型保存失败: {e}")
        logger.info(f"训练已终止: 会话 {session_id}")
        return session

    def get_session_status(self, session_id: UUID) -> TrainingSessionStatus | None:
        session = self._session_mgr.get_session(session_id)
        return session.status if session else None

    def get_active_session(self) -> TrainingSession | None:
        return self._session_mgr.get_active_session()

    @property
    def session_manager(self) -> TrainingSessionManager:
        return self._session_mgr

    def _run_training(self, session: TrainingSession, config: TrainingConfig, training_fn, callback):
        try:
            if training_fn is not None:
                training_fn(config, callback, pause_event, stop_event)
            else:
                logger.warning("未提供训练函数，训练将立即完成")
            if not stop_event.is_set():
                self._session_mgr.transition(session.session_id, TrainingSessionStatus.Completed)
                logger.info(f"训练完成: 会话 {session.session_id}")
        except Exception as e:
            logger.error(f"训练异常: {e}")
            try:
                self._session_mgr.transition(session.session_id, TrainingSessionStatus.Failed)
            except Exception:
                pass
