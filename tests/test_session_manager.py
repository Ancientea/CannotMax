from __future__ import annotations

import pytest

from gradio_ml.infrastructure.models import TrainingConfig, TrainingSessionStatus
from gradio_ml.infrastructure.session_manager import TrainingSessionManager


class TestSessionManagerStateTransitions:
    def test_idle_to_running(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        result = mgr.transition(session.session_id, TrainingSessionStatus.Running)
        assert result.status == TrainingSessionStatus.Running

    def test_running_to_paused(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        result = mgr.transition(session.session_id, TrainingSessionStatus.Paused)
        assert result.status == TrainingSessionStatus.Paused

    def test_paused_to_running(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        mgr.transition(session.session_id, TrainingSessionStatus.Paused)
        result = mgr.transition(session.session_id, TrainingSessionStatus.Running)
        assert result.status == TrainingSessionStatus.Running

    def test_running_to_completed(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        result = mgr.transition(session.session_id, TrainingSessionStatus.Completed)
        assert result.status == TrainingSessionStatus.Completed

    def test_running_to_terminated(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        result = mgr.transition(session.session_id, TrainingSessionStatus.Terminated)
        assert result.status == TrainingSessionStatus.Terminated

    def test_running_to_failed(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        result = mgr.transition(session.session_id, TrainingSessionStatus.Failed)
        assert result.status == TrainingSessionStatus.Failed

    def test_terminated_to_idle(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        mgr.transition(session.session_id, TrainingSessionStatus.Terminated)
        result = mgr.transition(session.session_id, TrainingSessionStatus.Idle)
        assert result.status == TrainingSessionStatus.Idle

    def test_invalid_transition_raises(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        with pytest.raises(RuntimeError, match="非法状态转换"):
            mgr.transition(session.session_id, TrainingSessionStatus.Paused)


class TestSessionManagerActiveSession:
    def test_get_active_session_running(self):
        mgr = TrainingSessionManager()
        session = mgr.create_session(TrainingConfig())
        mgr.transition(session.session_id, TrainingSessionStatus.Running)
        active = mgr.get_active_session()
        assert active is not None
        assert active.session_id == session.session_id

    def test_get_active_session_none_when_idle(self):
        mgr = TrainingSessionManager()
        assert mgr.get_active_session() is None

    def test_no_two_running_sessions(self):
        mgr = TrainingSessionManager()
        s1 = mgr.create_session(TrainingConfig())
        mgr.transition(s1.session_id, TrainingSessionStatus.Running)
        s2 = mgr.create_session(TrainingConfig())
        with pytest.raises(RuntimeError, match="已有训练会话正在运行"):
            mgr.transition(s2.session_id, TrainingSessionStatus.Running)
