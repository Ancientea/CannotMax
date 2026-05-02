"""Unit tests for ConnectorFactory state machine.

Tests the IDLE→VALID→INVALID state transitions, config change detection,
and automatic rebuild logic using Mock to avoid real device connections.
"""

from unittest.mock import Mock, patch

from src.cannotmax.core.connector.base_connector import BaseConnector

# Import under test
from src.cannotmax.core.connector.factory import ConnectorFactory, ConnectorState


class TestConnectorStateTransitions:
    """Test state machine transitions: IDLE→VALID→INVALID."""

    def setup_method(self):
        """Create fresh factory for each test."""
        self.factory = ConnectorFactory()

    def test_new_connector_starts_idle(self):
        """New connector should be created in IDLE state (no connection)."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_conn = Mock(spec=BaseConnector)
            mock_conn._controller = None  # Not connected
            MockAdb.return_value = mock_conn

            conn = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Verify created but not connected
            assert conn is not None
            assert "ADB" in self.factory._pool
            _, _, state = self.factory._pool["ADB"]
            assert state == ConnectorState.IDLE

            # Verify connect() was NOT called (truly lazy)
            mock_conn.connect.assert_not_called()

    def test_idle_to_valid_on_successful_local_check(self):
        """IDLE → VALID when local check passes (first use)."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_controller = Mock()
            mock_controller.is_alive.return_value = True

            mock_conn = Mock(spec=BaseConnector)
            mock_conn._controller = mock_controller
            MockAdb.return_value = mock_conn

            # First get: creates IDLE
            conn1 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")
            _, _, state1 = self.factory._pool["ADB"]
            assert state1 == ConnectorState.IDLE

            # Second get: patch local check to pass, should transition to VALID
            with patch.object(self.factory, "_is_local_usable", return_value=True):
                conn2 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            _, _, state2 = self.factory._pool["ADB"]
            assert state2 == ConnectorState.VALID
            assert conn1 is conn2  # Same instance reused

    def test_idle_to_invalid_on_failed_local_check(self):
        """IDLE → INVALID when local check fails (dead connection)."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            # First connector: dead controller
            dead_controller = Mock()
            dead_controller.is_alive.return_value = False
            dead_conn = Mock(spec=BaseConnector)
            dead_conn._controller = dead_controller

            # Second connector: fresh (for rebuild)
            fresh_controller = Mock()
            fresh_controller.is_alive.return_value = True
            fresh_conn = Mock(spec=BaseConnector)
            fresh_conn._controller = fresh_controller

            MockAdb.side_effect = [dead_conn, fresh_conn]

            # First get: creates IDLE
            conn1 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Second get: local check fails → rebuild
            conn2 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Should have rebuilt with new instance
            assert conn1 is not conn2
            _, _, state = self.factory._pool["ADB"]
            assert state == ConnectorState.IDLE  # New connector starts IDLE

            # Dead connector should be discarded
            dead_conn.disconnect.assert_called_once()

    def test_valid_stays_valid_without_check(self):
        """VALID state should be reused without any check (fast path)."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_controller = Mock()
            mock_controller.is_alive.return_value = True

            mock_conn = Mock(spec=BaseConnector)
            mock_conn._controller = mock_controller
            MockAdb.return_value = mock_conn

            # Create and mark as VALID manually (simulating successful first use)
            conn = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")
            self.factory._pool["ADB"] = (
                conn,
                {"adb_serial": "127.0.0.1:5555"},
                ConnectorState.VALID,
            )

            # Subsequent gets should reuse without checking is_alive
            conn2 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            assert conn is conn2
            # is_alive should NOT be called (no active testing!)
            mock_controller.is_alive.assert_not_called()

    def test_invalid_triggers_rebuild(self):
        """INVALID state should trigger immediate rebuild on next get()."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            # First connector (will be marked INVALID)
            invalid_conn = Mock(spec=BaseConnector)
            invalid_conn._controller = None

            # Second connector (rebuild)
            new_conn = Mock(spec=BaseConnector)
            new_conn._controller = Mock(is_alive=lambda: True)

            MockAdb.side_effect = [invalid_conn, new_conn]

            # Create first connector
            conn1 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Manually mark as INVALID (simulating operation failure)
            self.factory.mark_invalid("ADB")
            _, _, state = self.factory._pool["ADB"]
            assert state == ConnectorState.INVALID

            # Next get should rebuild
            conn2 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            assert conn1 is not conn2
            invalid_conn.disconnect.assert_called_once()


class TestConfigChangeDetection:
    """Test that config changes (e.g., ADB serial) trigger rebuild."""

    def setup_method(self):
        self.factory = ConnectorFactory()

    def test_same_config_reuses_connector(self):
        """Same kwargs should reuse existing connector."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_conn = Mock(spec=BaseConnector)
            MockAdb.return_value = mock_conn

            conn1 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Second get: local check passes → reuse without recreating
            with patch.object(self.factory, "_is_local_usable", return_value=True):
                conn2 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            assert conn1 is conn2
            MockAdb.assert_called_once()  # Only created once

    def test_different_serial_rebuilds_connector(self):
        """Different adb_serial should discard old and create new."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            conn1 = Mock(spec=BaseConnector)
            conn1._controller = Mock(is_alive=lambda: True)
            conn2 = Mock(spec=BaseConnector)
            conn2._controller = Mock(is_alive=lambda: True)
            MockAdb.side_effect = [conn1, conn2]

            # First serial
            c1 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")
            # Different serial
            c2 = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5556")

            assert c1 is not c2
            assert MockAdb.call_count == 2
            # Old connector should be disconnected
            conn1.disconnect.assert_called_once()

    def test_mode_switch_creates_separate_connectors(self):
        """Different modes should have separate connectors in pool."""
        with (
            patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb,
            patch("src.cannotmax.core.connector.factory.PcConnector") as MockPc,
        ):
            adb_conn = Mock(spec=BaseConnector)
            pc_conn = Mock(spec=BaseConnector)

            MockAdb.return_value = adb_conn
            MockPc.return_value = pc_conn

            adb = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")
            pc = self.factory.get_connector("PC")

            assert adb is not pc
            assert "ADB" in self.factory._pool
            assert "PC" in self.factory._pool


class TestLocalUsabilityCheck:
    """Test _is_local_usable() lightweight checks."""

    def setup_method(self):
        self.factory = ConnectorFactory()

    def test_adb_check_uses_is_alive(self):
        """AdbConnector local check should use controller.is_alive()."""
        from src.cannotmax.core.connector.adb_connector import AdbConnector

        mock_controller = Mock()
        mock_controller.is_alive.return_value = True

        # Create mock that passes isinstance check for AdbConnector
        mock_conn = Mock(spec=AdbConnector)
        mock_conn._controller = mock_controller

        # Test _is_local_usable directly
        result = self.factory._is_local_usable(mock_conn)

        assert result is True
        mock_controller.is_alive.assert_called_once()

    def test_adb_check_returns_false_when_controller_none(self):
        """Should return False if _controller is None (not initialized)."""
        mock_conn = Mock(spec=BaseConnector)
        mock_conn._controller = None

        result = self.factory._is_local_usable(mock_conn)
        assert result is False

    def test_adb_check_returns_false_when_is_alive_false(self):
        """Should return False if is_alive() returns False."""
        mock_controller = Mock()
        mock_controller.is_alive.return_value = False

        mock_conn = Mock(spec=BaseConnector)
        mock_conn._controller = mock_controller

        result = self.factory._is_local_usable(mock_conn)
        assert result is False

    def test_pc_check_uses_win32gui_iswindow(self):
        """PcConnector local check should use win32gui.IsWindow()."""
        from src.cannotmax.core.connector.pc_connector import PcConnector

        # Patch the module-level import in factory
        with patch("src.cannotmax.core.connector.factory.IsWindow") as mock_iswindow:
            mock_iswindow.return_value = True

            # Create mock that passes isinstance check for PcConnector
            mock_conn = Mock(spec=PcConnector)
            mock_conn._hwnd = 12345

            result = self.factory._is_local_usable(mock_conn)

            assert result is True
            mock_iswindow.assert_called_once_with(12345)

    def test_local_check_exception_returns_false(self):
        """Local check should return False on any exception."""
        mock_conn = Mock(spec=BaseConnector)
        mock_conn._controller = Mock()
        type(mock_conn._controller).is_alive = Mock(side_effect=Exception("Crash"))

        result = self.factory._is_local_usable(mock_conn)
        assert result is False


class TestPoolManagement:
    """Test pool operations: return, mark_invalid, disconnect_all."""

    def setup_method(self):
        self.factory = ConnectorFactory()

    def test_return_connector_marks_idle(self):
        """return_connector() should mark state as IDLE without testing."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_controller = Mock()
            mock_conn = Mock(spec=BaseConnector)
            mock_conn._controller = mock_controller
            MockAdb.return_value = mock_conn

            conn = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Manually set to VALID (simulating successful use)
            self.factory._pool["ADB"] = (
                conn,
                {"adb_serial": "127.0.0.1:5555"},
                ConnectorState.VALID,
            )

            # Return to pool
            self.factory.return_connector("ADB", conn)

            _, _, state = self.factory._pool["ADB"]
            assert state == ConnectorState.IDLE

            # No is_alive() check during return
            mock_controller.is_alive.assert_not_called()

    def test_mark_invalid_sets_state(self):
        """mark_invalid() should set state to INVALID."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_conn = Mock(spec=BaseConnector)
            MockAdb.return_value = mock_conn

            conn = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")
            self.factory.mark_invalid("ADB")

            _, _, state = self.factory._pool["ADB"]
            assert state == ConnectorState.INVALID

    def test_mark_valid_idempotent(self):
        """mark_valid() should only transition IDLE→VALID, not VALID→VALID or INVALID→VALID."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            mock_conn = Mock(spec=BaseConnector)
            MockAdb.return_value = mock_conn

            conn = self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # IDLE → VALID
            self.factory.mark_valid("ADB")
            assert self.factory._pool["ADB"][2] == ConnectorState.VALID

            # VALID stays VALID (idempotent)
            self.factory.mark_valid("ADB")
            assert self.factory._pool["ADB"][2] == ConnectorState.VALID

            # INVALID stays INVALID (no transition from mark_valid)
            self.factory.mark_invalid("ADB")
            self.factory.mark_valid("ADB")
            assert self.factory._pool["ADB"][2] == ConnectorState.INVALID

    def test_disconnect_all_clears_pool(self):
        """disconnect_all() should disconnect all connectors and clear pool."""
        with (
            patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb,
            patch("src.cannotmax.core.connector.factory.PcConnector") as MockPc,
        ):
            adb_conn = Mock(spec=BaseConnector)
            pc_conn = Mock(spec=BaseConnector)

            MockAdb.return_value = adb_conn
            MockPc.return_value = pc_conn

            self.factory.get_connector("ADB", adb_serial="127.0.0.1:5555")
            self.factory.get_connector("PC")

            assert len(self.factory._pool) == 2

            self.factory.disconnect_all()

            assert len(self.factory._pool) == 0
            adb_conn.disconnect.assert_called_once()
            pc_conn.disconnect.assert_called_once()


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_get_connector_invalid_mode(self):
        """Should return None for unknown mode (exception caught internally)."""
        factory = ConnectorFactory()

        result = factory.get_connector("INVALID_MODE")

        assert result is None
        assert "INVALID_MODE" not in factory._pool

    def test_get_connector_creation_exception(self):
        """Should return None if connector creation raises exception."""
        with patch(
            "src.cannotmax.core.connector.factory.AdbConnector",
            side_effect=Exception("Create failed"),
        ):
            factory = ConnectorFactory()

            result = factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            assert result is None
            assert "ADB" not in factory._pool

    def test_return_wrong_connector_ignores(self):
        """return_connector() should ignore if connector doesn't match pool."""
        with patch("src.cannotmax.core.connector.factory.AdbConnector") as MockAdb:
            conn1 = Mock(spec=BaseConnector)
            conn2 = Mock(spec=BaseConnector)
            MockAdb.return_value = conn1

            factory = ConnectorFactory()
            factory.get_connector("ADB", adb_serial="127.0.0.1:5555")

            # Try to return wrong connector (should be ignored)
            factory.return_connector("ADB", conn2)

            # State should remain unchanged
            assert "ADB" in factory._pool
