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
        self._pool: dict[str, tuple[BaseConnector, dict]] = {}  # (connector, kwargs)
    
    def get_connector(self, mode: str, **kwargs) -> Optional[BaseConnector]:
        """
        Get connector for mode, reusing if already connected.
        
        Args:
            mode: "ADB", "PC", or "WIN"
            **kwargs: Constructor args for the connector
        
        Returns:
            Connected connector or None if failed
        """
        # Reuse existing if connected and config matches
        if mode in self._pool:
            existing, existing_kwargs = self._pool[mode]
            
            # Check if config changed (e.g., different ADB serial)
            if existing_kwargs != kwargs:
                logger.info(f"{mode} config changed, recreating connector")
                try:
                    existing.disconnect()
                except Exception as e:
                    logger.warning(f"Disconnect failed: {e}")
                del self._pool[mode]
            elif existing.is_connected:
                logger.debug(f"Reusing existing {mode} connector")
                return existing
            else:
                # Try to reconnect first
                logger.info(f"Existing {mode} connector disconnected, attempting reconnect...")
                if existing.connect():
                    logger.debug(f"Reusing existing {mode} connector (reconnected)")
                    return existing
                else:
                    # Reconnect failed: disconnect and remove
                    try:
                        existing.disconnect()
                    except Exception as e:
                        logger.warning(f"Disconnect failed: {e}")
                    del self._pool[mode]
        
        # Create new (lazy connection, no blocking)
        logger.info(f"Creating new {mode} connector (not connected)")
        try:
            connector: BaseConnector = self._create_connector(mode, **kwargs)
            self._pool[mode] = (connector, kwargs)
            logger.debug(f"{mode} connector created, awaiting connection")
            return connector
        except Exception as e:
            logger.exception(f"{mode} creation exception: {e}")
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
        for mode, (conn, _) in list(self._pool.items()):
            try:
                conn.disconnect()
            except Exception as e:
                logger.warning(f"Disconnect {mode} failed: {e}")
        self._pool.clear()
