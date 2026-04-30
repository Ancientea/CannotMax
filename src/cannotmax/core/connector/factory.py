"""Connector factory for lifecycle management and pooling."""
import logging
from typing import Optional
from .base_connector import BaseConnector
from .adb_connector import AdbConnector
from .pc_connector import PcConnector

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Manages Connector lifecycle with per-mode singleton pooling."""
    
    def __init__(self):
        self._pool: dict[str, BaseConnector] = {}
    
    def get_connector(self, mode: str, **kwargs) -> Optional[BaseConnector]:
        """
        Get connector for mode, reusing if already connected.
        
        Args:
            mode: "ADB", "PC", or "WIN"
            **kwargs: Constructor args for the connector
        
        Returns:
            Connected connector or None if failed
        """
        # Reuse existing if connected
        if mode in self._pool:
            existing = self._pool[mode]
            if existing.is_connected:
                logger.debug(f"Reusing existing {mode} connector")
                return existing
            else:
                # Disconnect failed instance
                try:
                    existing.disconnect()
                except Exception as e:
                    logger.warning(f"Disconnect failed: {e}")
                del self._pool[mode]
        
        # Create new
        logger.info(f"Creating new {mode} connector")
        try:
            connector: BaseConnector = self._create_connector(mode, **kwargs)
            success = connector.connect()
            
            if success:
                self._pool[mode] = connector
                logger.info(f"{mode} connected successfully")
                return connector
            else:
                logger.warning(f"{mode} connection failed")
                return None
                
        except Exception as e:
            logger.exception(f"{mode} connection exception: {e}")
            return None
    
    def _create_connector(self, mode: str, **kwargs) -> BaseConnector:
        """Create connector instance."""
        if mode == "ADB":
            return AdbConnector(**kwargs)
        elif mode in ("PC", "WIN"):
            return PcConnector(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def disconnect_all(self):
        """Disconnect all connectors in pool."""
        for mode, conn in list(self._pool.items()):
            try:
                conn.disconnect()
            except Exception as e:
                logger.warning(f"Disconnect {mode} failed: {e}")
        self._pool.clear()
