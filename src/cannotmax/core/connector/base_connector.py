from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

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
    
    @abstractmethod
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """
        Capture screenshot as BGR numpy array.
        
        Returns:
            np.ndarray: BGR image (H, W, 3), None if failed
        """
        pass
    
    @abstractmethod
    def click(self, point: tuple[float, float]) -> None:
        """
        Click at relative coordinates (0-1).
        
        Args:
            point: (x, y) where x,y in [0, 1]
        """
        pass
    
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
