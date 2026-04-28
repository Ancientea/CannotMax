from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID, uuid4

from gradio_ml.infrastructure.models import TrainingConfig, TrainingSession, TrainingSessionStatus

logger = logging.getLogger(__name__)

_VALID_TRANSITIONS: dict[TrainingSessionStatus, set[TrainingSessionStatus]] = {
    TrainingSessionStatus.Idle: {TrainingSessionStatus.Running},
    TrainingSessionStatus.Running: {TrainingSessionStatus.Paused, TrainingSessionStatus.Completed, TrainingSessionStatus.Terminated, TrainingSessionStatus.Failed},
    TrainingSessionStatus.Paused: {TrainingSessionStatus.Running, TrainingSessionStatus.Terminated},
    TrainingSessionStatus.Completed: {TrainingSessionStatus.Idle},
    TrainingSessionStatus.Terminated: {TrainingSessionStatus.Idle},
    TrainingSessionStatus.Failed: {TrainingSessionStatus.Idle},
}


class TrainingSessionManager:
    def __init__(self):
        self._sessions: dict[UUID, TrainingSession] = {}
        self._active_session_id: UUID | None = None

    def create_session(self, config: TrainingConfig) -> TrainingSession:
        session = TrainingSession(
            session_id=uuid4(),
            config=config,
            status=TrainingSessionStatus.Idle,
            created_at=datetime.now(),
        )
        self._sessions[session.session_id] = session
        return session

    def transition(self, session_id: UUID, new_status: TrainingSessionStatus) -> TrainingSession:
        session = self._get_session(session_id)
        if new_status not in _VALID_TRANSITIONS.get(session.status, set()):
            raise RuntimeError(
                f"非法状态转换: {session.status.value} -> {new_status.value}"
            )
        if new_status == TrainingSessionStatus.Running:
            active = self.get_active_session()
            if active is not None and active.session_id != session_id and active.status == TrainingSessionStatus.Running:
                raise RuntimeError("已有训练会话正在运行，无法启动新训练")
            self._active_session_id = session_id
        elif new_status == TrainingSessionStatus.Paused:
            pass
        elif new_status in (TrainingSessionStatus.Completed, TrainingSessionStatus.Terminated, TrainingSessionStatus.Failed):
            pass
        elif new_status == TrainingSessionStatus.Idle:
            self._active_session_id = None

        session.status = new_status
        logger.info(f"会话 {session_id} 状态转换: -> {new_status.value}")
        return session

    def get_session(self, session_id: UUID) -> TrainingSession | None:
        return self._sessions.get(session_id)

    def get_active_session(self) -> TrainingSession | None:
        if self._active_session_id is None:
            return None
        session = self._sessions.get(self._active_session_id)
        if session is None or session.status in (TrainingSessionStatus.Idle, TrainingSessionStatus.Completed, TrainingSessionStatus.Terminated, TrainingSessionStatus.Failed):
            self._active_session_id = None
            return None
        return session

    def update_progress(self, session_id: UUID, epoch: int, step: int) -> None:
        session = self._get_session(session_id)
        session.current_epoch = epoch
        session.current_step = step
        session.last_epoch_callback_time = datetime.now()

    def _get_session(self, session_id: UUID) -> TrainingSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"会话不存在: {session_id}")
        return session
