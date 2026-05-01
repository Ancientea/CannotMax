"""Connector factory with state-based lazy pooling.

Uses state marking (IDLE/VALID/INVALID) to avoid active connection testing,
solving the lazy connection vs. instance pooling contradiction.
"""

import logging
from enum import Enum
from typing import Optional
from win32gui import IsWindow  # Module-level import for performance
from .base_connector import BaseConnector
from .adb_connector import AdbConnector
from .pc_connector import PcConnector

logger = logging.getLogger(__name__)


class ConnectorState(Enum):
    """Connection state machine for lazy validation."""

    IDLE = 0  # 刚创建/归还，未验证（默认状态）
    VALID = 1  # 已验证可用（首次操作成功）
    INVALID = 2  # 已知失效（需重建）


class ConnectorFactory:
    """Manages Connector lifecycle with state-based lazy pooling.

    Key design:
    - No active connection testing (no ping/select)
    - State marking instead of live checks
    - Only verifies when borrowing IDLE connections
    - Automatic rebuild on failure (transparent to caller)
    """

    def __init__(self):
        # Pool stores: {mode: (connector, kwargs, state)}
        self._pool: dict[str, tuple[BaseConnector, dict, ConnectorState]] = {}

    def get_connector(self, mode: str, **kwargs) -> Optional[BaseConnector]:
        """
        Get connector with lazy validation (no active testing!).

        State transitions:
        - IDLE → VALID: First use, light local check passes
        - IDLE → INVALID: Local check fails (rebuild immediately)
        - VALID → IDLE: Returned to pool (no testing)
        - INVALID → [*]: Rebuild on next get()

        Args:
            mode: "ADB", "PC", or "WIN"
            **kwargs: Constructor args

        Returns:
            Connector instance (may be unconnected if IDLE, but usable)
        """
        # 1. Check config match
        if mode in self._pool:
            existing, existing_kwargs, state = self._pool[mode]

            if existing_kwargs != kwargs:
                logger.info("%s config changed, discarding connector", mode)
                self._discard_connector(existing)
                del self._pool[mode]
            else:
                # 2. State-based decision (NO ACTIVE TESTING!)
                if state == ConnectorState.VALID:
                    logger.debug("Reusing VALID %s connector (no check)", mode)
                    return existing

                elif state == ConnectorState.INVALID:
                    logger.info("%s connector INVALID, rebuilding...", mode)
                    self._discard_connector(existing)
                    del self._pool[mode]

                else:  # IDLE
                    # 3. Light local check (0.1μs, no network IO!)
                    logger.debug("%s connector IDLE, checking local state...", mode)
                    if self._is_local_usable(existing):
                        logger.debug(
                            "%s connector passed local check, marking VALID", mode
                        )
                        self._pool[mode] = (existing, kwargs, ConnectorState.VALID)
                        return existing
                    else:
                        logger.info(
                            "%s connector failed local check (idle), rebuilding...",
                            mode,
                        )
                        self._discard_connector(existing)
                        del self._pool[mode]

        # 4. Create new (truly lazy, no connection!)
        logger.info("Creating new %s connector (IDLE state)", mode)
        try:
            connector = self._create_connector(mode, **kwargs)
            self._pool[mode] = (connector, kwargs, ConnectorState.IDLE)
            logger.debug("%s connector created (IDLE)", mode)
            return connector
        except Exception as e:
            logger.exception(f"{mode} creation exception: {e}")
            return None

    def return_connector(self, mode: str, connector: BaseConnector):
        """Return connector to pool (mark IDLE, NO TESTING!).

        Args:
            mode: Mode key
            connector: Connector to return
        """
        if mode in self._pool:
            _, kwargs, _ = self._pool[mode]
            if self._pool[mode][0] is connector:
                # Simply mark IDLE (0 overhead!)
                self._pool[mode] = (connector, kwargs, ConnectorState.IDLE)
                logger.debug("Returned %s connector to pool (IDLE)", mode)

    def mark_invalid(self, mode: str):
        """Mark connector as invalid (call from error handlers).

        Used when operation fails with connection error.
        """
        if mode in self._pool:
            conn, kwargs, _ = self._pool[mode]
            self._pool[mode] = (conn, kwargs, ConnectorState.INVALID)
            logger.debug("Marked %s connector INVALID", mode)

    def mark_valid(self, mode: str):
        """Mark connector as valid (call after successful operation).

        Used when IDLE connector passes first use (e.g., successful screenshot).
        Only marks VALID if currently IDLE (idempotent).
        """
        if mode in self._pool:
            conn, kwargs, state = self._pool[mode]
            if state == ConnectorState.IDLE:
                self._pool[mode] = (conn, kwargs, ConnectorState.VALID)
                logger.debug("Marked %s connector VALID (IDLE→VALID)", mode)
            elif state != ConnectorState.VALID:
                logger.debug("Attempted to mark %s VALID, but state is %s", mode, state)

    def _is_local_usable(self, connector: BaseConnector) -> bool:
        """Lightweight local check (NO NETWORK IO!).

        Checks OS-level state (socket closed, window valid), not network connectivity.
        Must be <1ms to preserve lazy connection benefit.

        Returns:
            True if connector appears usable locally
        """
        try:
            if isinstance(connector, AdbConnector):
                # MAA Framework: check if controller exists and socket not closed
                if (
                    not hasattr(connector, "_controller")
                    or connector._controller is None
                ):
                    return False
                # Check MAA AdbController alive status (local, non-blocking)
                if hasattr(connector._controller, "is_alive"):
                    return connector._controller.is_alive()
                # Fallback: check if _connected flag exists
                return getattr(connector, "_connected", False)

            elif isinstance(connector, PcConnector):
                # PC Mode: check window handle validity (Windows API, local)
                if not hasattr(connector, "_hwnd") or connector._hwnd is None:
                    return False
                # IsWindow is local API call, <1ms
                return IsWindow(connector._hwnd)

            else:
                # Generic: check _connected flag
                return getattr(connector, "_connected", False)

        except Exception as e:
            logger.debug("Local check exception: %s", e)
            return False

    def _discard_connector(self, connector: BaseConnector):
        """Safely discard connector."""
        try:
            connector.disconnect()
        except Exception as e:
            logger.warning("Discard disconnect failed: %s", e)

    def _create_connector(self, mode: str, **kwargs) -> BaseConnector:
        """Create connector instance (not connected)."""
        if mode == "ADB":
            return AdbConnector(**kwargs)
        elif mode in ("PC", "WIN"):
            return PcConnector(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def disconnect_all(self):
        """Disconnect all connectors."""
        for mode, (conn, _, _) in list(self._pool.items()):
            try:
                conn.disconnect()
            except Exception as e:
                logger.warning("Disconnect %s failed: %s", mode, e)
        self._pool.clear()
