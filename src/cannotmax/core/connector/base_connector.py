import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base class for device connectors.

    Provides unified interface for screenshot capture and input simulation
    across different platforms (Android/PC).
    """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connector is active."""
        pass

    @property
    @abstractmethod
    def screen_width(self) -> int:
        """Get screen width."""
        pass

    @property
    @abstractmethod
    def screen_height(self) -> int:
        """Get screen height."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True if successful."""
        pass

    @abstractmethod
    def ensure_connected(self, max_retries: int = 3) -> bool:
        """Ensure connection is active. Auto-connect if needed with retry.

        Args:
            max_retries: Maximum connection attempts before giving up (default: 3)

        Returns:
            bool: True if connected after call, False if all retries failed.
        """
        pass

    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot with auto-connect."""
        if not self.ensure_connected():
            logger.warning("Cannot capture: device not connected")
            return None
        return self._capture_internal()

    def click(self, point: tuple[float, float]) -> bool:
        """Click with auto-connect. Returns True if successful."""
        if not self.ensure_connected():
            logger.warning("Cannot click: device not connected")
            return False
        self._click_internal(point)
        return True

    @abstractmethod
    def get_device_list(self) -> list[str]:
        """Get available device identifiers."""
        pass

    @abstractmethod
    def update_device_serial(self, serial: str) -> str:
        """Update device serial/connection string."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect and cleanup resources."""
        pass

    @abstractmethod
    def _capture_internal(self) -> Optional[np.ndarray]:
        """Actual capture logic, assumes connected."""
        pass

    @abstractmethod
    def _click_internal(self, point: tuple[float, float]) -> None:
        """Actual click logic, assumes connected."""
        pass
