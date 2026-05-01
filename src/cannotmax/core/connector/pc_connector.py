"""PC connector with MAA Framework support.

Connects to Arknights PC client via Windows API.
Automatically uses MAA Framework Win32Controller if available,
otherwise falls back to WinRT screen capture + SendInput.

Usage:
    conn = PcConnector(window_name="明日方舟")
    conn.connect()
    img = conn.capture_screenshot()  # MAA or WinRT
    conn.click((0.5, 0.5))  # MAA click or SendInput
"""

import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import win32gui
import win32con
import ctypes

from .base_connector import BaseConnector
from .winrt_capture import WinRTScreenCapture

logger = logging.getLogger(__name__)


class PcConnector(BaseConnector):
    """Windows PC connector for Arknights client.

    Features:
    - Auto-detects MAA Framework availability
    - MAA mode: FramePool screencap + SendMessageWithCursorPos (background)
    - Legacy mode: WinRT capture + SendInput (foreground required)
    """

    def __init__(self, window_name: str = "明日方舟"):
        self._window_name = window_name
        self._hwnd = None
        self._screen_width = 0
        self._screen_height = 0
        self._is_connected = False
        self._maa_controller = None
        self._maa_available = False
        self._winrt_capture = None

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

    def _find_all_windows(self, pattern: str) -> list[int]:
        """Enumerate all visible windows with title containing pattern."""
        matches = []

        def enum_proc(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if pattern in title:
                    matches.append(hwnd)
            return True

        win32gui.EnumWindows(enum_proc, 0)
        return matches

    def _select_window(self, hwnds: list[int]) -> Optional[int]:
        """Show window picker dialog limited to given hwnds."""
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if not app:
            logger.error("QApplication not available for window selection")
            return None

        from ...gui.dialogs.window_picker import WindowPickerDialog

        parent = app.activeWindow() if hasattr(app, "activeWindow") else None
        dlg = WindowPickerDialog(parent, filter_hwnds=hwnds)

        if dlg.exec():
            sel = dlg.get_selection()
            if sel and "hwnd" in sel:
                return sel["hwnd"]
        return None

    def connect(self) -> bool:
        """Connect to PC window with multi-window detection."""
        # 1. Find all matching windows
        hwnds = self._find_all_windows(self._window_name)

        if not hwnds:
            logger.error(f"No windows found matching: {self._window_name}")
            return False

        # 2. Select if multiple
        if len(hwnds) == 1:
            self._hwnd = hwnds[0]
            logger.info(f"Auto-selected window: {self._hwnd}")
        else:
            logger.info(f"Found {len(hwnds)} windows, showing selector")
            selected = self._select_window(hwnds)
            if selected is None:
                logger.info("User cancelled window selection")
                return False
            self._hwnd = selected

        # 3. Get resolution
        rect = win32gui.GetClientRect(self._hwnd)
        self._screen_width = rect[2] - rect[0]
        self._screen_height = rect[3] - rect[1]

        # 4. Initialize MAA or WinRT
        self._init_maa()
        if not self._maa_available:
            self._init_winrt()

        self._is_connected = True
        logger.info(
            f"PC connected: {self._window_name}, "
            f"hwnd={self._hwnd}, {self._screen_width}x{self._screen_height}, "
            f"MAA={'enabled' if self._maa_available else 'disabled'}"
        )
        return True

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
        """Initialize MAA Win32Controller if available."""
        from ...config import DISABLE_MAAFW

        if DISABLE_MAAFW:
            logger.info("MAA disabled by config (control.disable_maafw)")
            self._maa_available = False
            return
        maa_controller = None
        try:
            from maa.toolkit import Toolkit
            from maa.controller import (
                Win32Controller,
                MaaWin32ScreencapMethodEnum,
                MaaWin32InputMethodEnum,
            )

            Toolkit.init_option(str(Path.cwd()))

            # Use Seize (foreground) mode — Arknights PC uses ACE + Raw Input
            maa_controller = Win32Controller(
                self._hwnd,
                screencap_method=MaaWin32ScreencapMethodEnum.FramePool,
                mouse_method=MaaWin32InputMethodEnum.Seize,
                keyboard_method=MaaWin32InputMethodEnum.Seize,
            )
            maa_controller.post_connection().wait()
            maa_controller.set_screenshot_use_raw_size(True)

            # Only set instance variable after successful initialization
            self._maa_controller = maa_controller
            maa_controller = None  # Prevent __del__ from running on temporary object
            self._maa_available = True
            logger.info("MAA Win32Controller initialized")

        except Exception as e:
            logger.warning(f"MAA unavailable, using WinRT + SendInput: {e}")
            self._maa_controller = None
            self._maa_available = False
        finally:
            # Ensure temporary object is cleaned up if creation failed
            if maa_controller is not None:
                del maa_controller

    def _init_winrt(self):
        """Initialize WinRT screen capture."""
        try:
            self._winrt_capture = WinRTScreenCapture(window_name=self._window_name)
            self._winrt_capture.start()
            logger.info("WinRT capture initialized")
        except Exception as e:
            logger.error(f"WinRT initialization failed: {e}")
            self._winrt_capture = None

    def _capture_internal(self) -> Optional[np.ndarray]:
        """Capture screenshot using MAA (preferred) or WinRT."""
        if not self._is_connected:
            return None

        if self._maa_available and self._maa_controller:
            img = self._capture_maa()
        else:
            img = self._capture_winrt()

        return img

    def _capture_maa(self) -> Optional[np.ndarray]:
        """Capture using MAA FramePool."""
        try:
            self._maa_controller.post_screencap().wait()
            return self._maa_controller.cached_image
        except Exception as e:
            logger.error(f"MAA capture failed: {e}")
            return self._capture_winrt()

    def _capture_winrt(self) -> Optional[np.ndarray]:
        """Capture using WinRT."""
        if not self._winrt_capture:
            return None
        try:
            return self._winrt_capture.snapshot()
        except Exception as e:
            logger.error(f"WinRT capture failed: {e}")
            return None

    def _click_internal(self, point: tuple[float, float]) -> None:
        """Click using MAA (preferred) or SendInput."""
        if not self._is_connected:
            return

        # Refresh client rect
        rect = win32gui.GetClientRect(self._hwnd)
        self._screen_width = rect[2] - rect[0]
        self._screen_height = rect[3] - rect[1]

        x, y = point
        x_coord = int(x * self._screen_width)
        y_coord = int(y * self._screen_height)

        if self._maa_available and self._maa_controller:
            self._click_maa(x_coord, y_coord)
        else:
            self._click_sendinput(x_coord, y_coord)

    def _click_maa(self, x: int, y: int) -> None:
        """Click using MAA SendMessageWithCursorPos."""
        try:
            logger.info(f"MAA click: ({x}, {y})")
            self._maa_controller.post_click(x, y).wait()
        except Exception as e:
            logger.error(f"MAA click failed: {e}")
            self._click_sendinput(x, y)

    def _click_sendinput(self, x: int, y: int) -> None:
        """Click using SendInput (requires foreground window)."""
        try:
            # Convert client coords to screen coords
            client_left, client_top = win32gui.ClientToScreen(self._hwnd, (0, 0))
            screen_x = client_left + x
            screen_y = client_top + y

            logger.info(
                f"SendInput click: window({x}, {y}) -> screen({screen_x}, {screen_y})"
            )

            # Try to bring to foreground
            try:
                if win32gui.GetForegroundWindow() != self._hwnd:
                    win32gui.ShowWindow(self._hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(self._hwnd)
                    time.sleep(0.1)
            except Exception:
                pass

            # SendInput click
            self._send_mouse_click(screen_x, screen_y)

        except Exception as e:
            logger.exception(f"SendInput click failed: {e}")

    def _send_mouse_click(self, screen_x: int, screen_y: int) -> None:
        """Send mouse click via SendInput at virtual desktop coordinates."""
        # Get virtual desktop metrics
        SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
        SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79

        vs_x = ctypes.windll.user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vs_y = ctypes.windll.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vs_w = ctypes.windll.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        vs_h = ctypes.windll.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

        if vs_w == 0:
            vs_w, vs_h = 1920, 1080

        # Convert to 0-65535 virtual coordinates
        dx = int((screen_x - vs_x) * 65535 / vs_w)
        dy = int((screen_y - vs_y) * 65535 / vs_h)

        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_ABSOLUTE = 0x8000
        MOUSEEVENTF_VIRTUALDESK = 0x4000
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004

        PUL = ctypes.POINTER(ctypes.c_ulong)

        class MouseInput(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL),
            ]

        class Input_I(ctypes.Union):
            _fields_ = [("mi", MouseInput)]

        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

        extra = ctypes.c_ulong(0)
        ii_ = Input_I()

        # Move
        ii_.mi = MouseInput(
            dx,
            dy,
            0,
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            0,
            ctypes.pointer(extra),
        )
        cmd = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
        time.sleep(0.05)

        # Down
        ii_.mi = MouseInput(
            dx,
            dy,
            0,
            MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            0,
            ctypes.pointer(extra),
        )
        cmd = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))
        time.sleep(0.05)

        # Up
        ii_.mi = MouseInput(
            dx,
            dy,
            0,
            MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            0,
            ctypes.pointer(extra),
        )
        cmd = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(cmd), ctypes.sizeof(cmd))

    def get_device_list(self) -> list[str]:
        """Check if window exists."""
        hwnd = win32gui.FindWindow(None, self._window_name)
        if hwnd:
            return [f"PC: {self._window_name}"]
        return []

    def update_device_serial(self, serial: str) -> str:
        """No-op for PC connector."""
        return self._window_name

    def disconnect(self) -> None:
        """Disconnect PC connector and cleanup resources."""
        try:
            if self._maa_controller is not None:
                del self._maa_controller
                self._maa_controller = None
            self._maa_available = False

            if self._winrt_capture is not None:
                self._winrt_capture.stop()
                self._winrt_capture = None

            self._is_connected = False
            self._hwnd = None
            logger.info(f"PC disconnected: {self._window_name}")
        except Exception as e:
            logger.warning(f"PC disconnect error: {e}")
