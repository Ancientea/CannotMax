"""ADB connector with MAA Framework support.

Connects to Android emulators (LDPlayer, MuMu, BlueStacks, Nox) via ADB.
Automatically uses MAA Framework if available, otherwise falls back to
raw ADB commands (screencap, input tap).

Usage:
    conn = AdbConnector(serial="127.0.0.1:5555")
    conn.connect()
    img = conn.capture_screenshot()  # MAA or ADB screencap
    conn.click((0.5, 0.5))  # MAA click or ADB input tap
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base_connector import BaseConnector

logger = logging.getLogger(__name__)


class AdbConnector(BaseConnector):
    """ADB-based connector for Android emulators.

    Features:
    - Auto-detects MAA Framework availability
    - MAA mode: High-performance raw screencap + MAA click (background)
    - Legacy mode: ADB screencap -p + input tap commands
    """

    def __init__(self, adb_serial: Optional[str] = None):
        self._adb_path = Path(r".\3rdparty\platform-tools\adb.exe").resolve()
        self._device_serial = adb_serial or "127.0.0.1:5555"
        self._screen_width = 0
        self._screen_height = 0
        self._is_connected = False
        self._maa_controller = None
        self._maa_available = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def screen_width(self) -> int:
        return self._screen_width

    @property
    def screen_height(self) -> int:
        return self._screen_height

    @property
    def is_maa_available(self) -> bool:
        """Check if MAA Framework is available and initialized."""
        return self._maa_available

    @property
    def device_serial(self) -> str:
        """Get the device serial number."""
        return self._device_serial

    def connect(self) -> bool:
        """Connect to ADB device and initialize MAA if available."""
        try:
            # Update device serial
            self.update_device_serial(self._device_serial)
            if not self._device_serial:
                return False

            # Get resolution
            self._screen_width, self._screen_height = self._get_window_size()
            self._is_connected = True

            # Try to initialize MAA Framework
            self._init_maa()

            logger.info(
                "ADB connected: %s, %sx%s, MAA=%s",
                self._device_serial,
                self._screen_width,
                self._screen_height,
                "enabled" if self._maa_available else "disabled",
            )
            return True

        except Exception as e:
            logger.exception(f"ADB connection failed: {e}")
            self._is_connected = False
            return False

    def ensure_connected(self, max_retries: int = 3) -> bool:
        """Ensure connection with retry logic."""
        if self._is_connected:
            return True

        for attempt in range(max_retries):
            try:
                if self.connect():
                    return True
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 500ms delay between retries
            except Exception as e:
                logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        logger.error(f"Failed to connect after {max_retries} attempts")
        return False

    def _init_maa(self):
        """Initialize MAA Framework controller if available."""
        from cannotmax.config import DISABLE_MAAFW

        if DISABLE_MAAFW:
            logger.info("MAA disabled by config (control.disable_maafw)")
            self._maa_available = False
            return
        maa_controller = None
        try:
            from maa.controller import AdbController
            from maa.toolkit import Toolkit

            Toolkit.init_option(str(Path.cwd()))

            maa_controller = AdbController(
                address=self._device_serial,
                adb_path=str(self._adb_path),
            )
            maa_controller.post_connection().wait()

            # Only set instance variable after successful initialization
            self._maa_controller = maa_controller
            maa_controller = None  # Prevent __del__ from running on temporary object
            self._maa_available = True
            logger.info("MAA Framework ADB initialized")

        except Exception as e:
            logger.warning(f"MAA Framework unavailable, using legacy ADB: {e}")
            self._maa_controller = None
            self._maa_available = False
        finally:
            # Ensure temporary object is cleaned up if creation failed
            if maa_controller is not None:
                del maa_controller

    def _get_window_size(self) -> tuple[int, int]:
        """Get screen resolution via ADB wm size."""
        try:
            cmd = f"{self._adb_path} -s {self._device_serial} shell wm size"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=True
            )
            output = result.stdout.strip()

            if "Physical size:" in output:
                res_str = output.split("Physical size:")[1].strip()
            elif "Override size:" in output:
                res_str = output.split("Override size:")[1].strip()
            else:
                raise ValueError("Cannot parse wm size output")

            width, height = map(int, res_str.split("x"))
            # Landscape mode
            if width < height:
                width, height = height, width
            return width, height

        except Exception:
            logger.warning("Failed to get resolution, using default 1920x1080")
            return 1920, 1080

    def _capture_internal(self) -> Optional[np.ndarray]:
        """Capture screenshot using MAA (preferred) or legacy ADB."""
        if not self._is_connected:
            return None

        if self._maa_available and self._maa_controller:
            return self._capture_maa()
        return self._capture_legacy()

    def _capture_maa(self) -> Optional[np.ndarray]:
        """Capture using MAA Framework (fast, raw format)."""
        try:
            self._maa_controller.post_screencap().wait()
            return self._maa_controller.cached_image
        except Exception as e:
            logger.error(f"MAA screencap failed: {e}")
            # Fallback to legacy
            return self._capture_legacy()

    def _capture_legacy(self) -> Optional[np.ndarray]:
        """Capture using ADB screencap -p."""
        try:
            cmd = f"{self._adb_path} -s {self._device_serial} exec-out screencap -p"
            screenshot_data = subprocess.check_output(cmd, shell=True)
            img_array = np.frombuffer(screenshot_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Cannot decode image")
            return img
        except Exception as e:
            logger.exception(f"Legacy screencap failed: {e}")
            return None

    def _click_internal(self, point: tuple[float, float]) -> None:
        """Click using MAA (preferred) or legacy ADB input tap."""
        if not self._is_connected:
            return

        x, y = point
        x_coord = int(x * self._screen_width)
        y_coord = int(y * self._screen_height)

        if self._maa_available and self._maa_controller:
            self._click_maa(x_coord, y_coord)
        else:
            self._click_legacy(x_coord, y_coord)

    def _click_maa(self, x: int, y: int) -> None:
        """Click using MAA Framework (background operation)."""
        try:
            logger.info(f"MAA click: ({x}, {y})")
            self._maa_controller.post_click(x, y).wait()
        except Exception as e:
            logger.error(f"MAA click failed: {e}")
            # Fallback to legacy
            self._click_legacy(x, y)

    def _click_legacy(self, x: int, y: int) -> None:
        """Click using ADB input tap."""
        try:
            logger.info(f"ADB click: ({x}, {y})")
            cmd = f"{self._adb_path} -s {self._device_serial} shell input tap {x} {y}"
            subprocess.run(cmd, shell=True)
        except Exception as e:
            logger.exception(f"ADB click failed: {e}")

    def get_device_list(self) -> list[str]:
        """Get list of connected ADB devices."""
        try:
            cmd = f"{self._adb_path} devices"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=2
            )
            devices = []
            for line in result.stdout.split("\n"):
                if "\tdevice" in line:
                    devices.append(line.split("\t")[0])
            return devices
        except subprocess.TimeoutExpired:
            logger.warning("Get device list timed out")
            return []
        except FileNotFoundError:
            logger.warning("ADB executable not found")
            return []
        except Exception as e:
            logger.exception(f"Get device list failed: {e}")
            return []

    def update_device_serial(self, serial: str) -> str:
        """Update ADB device serial and reconnect."""
        try:
            if not serial:
                serial = "127.0.0.1:5555"

            # Connect
            cmd = f"{self._adb_path} connect {serial}"
            subprocess.run(cmd, shell=True, check=True)

            # Verify
            devices = self.get_device_list()
            if serial in devices:
                logger.info(f"Using device: {serial}")
                self._device_serial = serial
                return serial
            elif devices:
                self._device_serial = devices[0]
                logger.info(f"Auto-selected: {self._device_serial}")
                return self._device_serial
            else:
                logger.error("No devices found")
                self._device_serial = ""
                return ""

        except Exception as e:
            logger.exception(f"Update device serial failed: {e}")
            self._device_serial = ""
            return ""

    def disconnect(self) -> None:
        """Disconnect ADB and cleanup MAA controller."""
        try:
            if self._maa_controller is not None:
                del self._maa_controller
                self._maa_controller = None
            self._maa_available = False
            self._is_connected = False
            self._stop_adb_server()
            logger.info(f"ADB disconnected: {self._device_serial}")
        except Exception as e:
            logger.warning(f"ADB disconnect error: {e}")

    def _stop_adb_server(self) -> None:
        """Kill ADB server to clean up stale connections."""
        if not self._adb_path.exists():
            return
        try:
            subprocess.run(
                [str(self._adb_path), "kill-server"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            logger.info("ADB server stopped")
        except Exception as e:
            logger.warning(f"Failed to stop ADB server: {e}")
