"""Unit tests for AutoFetch state machine with mock connector."""

import pytest
from src.cannotmax.core.auto_fetch import AutoFetch, GameState
from tests.mock_connector import MockConnector


@pytest.fixture
def fetcher():
    conn = MockConnector()
    af = AutoFetch(
        connector=conn,
        game_mode="单人",
        is_invest=False,
        update_prediction_callback=lambda v: None,
        update_monster_callback=lambda v: None,
        updater=lambda: None,
        start_callback=lambda: None,
        stop_callback=lambda: None,
        training_duration=3600,
        recognizer=None,
        cannot_model=None,
        capture_mode="ADB",
    )
    return af


class TestAutoFetchInit:
    def test_initial_state_is_unknown(self, fetcher):
        assert fetcher.last_state == GameState.UNKNOWN

    def test_not_running_on_init(self, fetcher):
        assert not fetcher.auto_fetch_running

    def test_stores_capture_mode(self, fetcher):
        assert fetcher.capture_mode == "ADB"


class TestAutoFetchLifecycle:
    def test_start_sets_running_flag(self, fetcher):
        fetcher.start_auto_fetch()
        try:
            assert fetcher.auto_fetch_running
        finally:
            fetcher.stop_auto_fetch()

    def test_stop_clears_running_flag(self, fetcher):
        fetcher.start_auto_fetch()
        fetcher.stop_auto_fetch()
        assert not fetcher.auto_fetch_running

    def test_double_stop_is_safe(self, fetcher):
        fetcher.start_auto_fetch()
        fetcher.stop_auto_fetch()
        fetcher.stop_auto_fetch()  # should not raise
        assert not fetcher.auto_fetch_running
