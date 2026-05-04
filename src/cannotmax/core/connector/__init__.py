"""Connector base interface for device communication.

Abstract base class defining the common interface for:
- AdbConnector: Android emulator/ADB connections
- PcConnector: Windows PC client connections

Both connectors automatically detect MAA Framework availability:
- If MAA available: Use MAA for screencap/click (background operation)
- If MAA unavailable: Use legacy implementation (ADB/Win32 API)
"""

from .adb_connector import AdbConnector
from .base_connector import BaseConnector
from .factory import ConnectorFactory
from .pc_connector import PcConnector

__all__ = [
    "BaseConnector",
    "AdbConnector",
    "PcConnector",
    "ConnectorFactory",
]
