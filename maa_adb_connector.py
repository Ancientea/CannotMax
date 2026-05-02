import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def resolve_maafw_path() -> str:
    if os.environ.get("MAAFW_BINARY_PATH"):
        return os.environ["MAAFW_BINARY_PATH"]
    candidates = [
        Path(sys.executable).parent / "maafw",
        Path.cwd() / "maafw",
        Path(__file__).resolve().parent / "maafw",
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob("MaaFramework.dll")):
            resolved = str(p)
            os.environ["MAAFW_BINARY_PATH"] = resolved
            logger.info(f"自动解析maafw路径: {resolved}")
            return resolved
    return ""


class MaaAvailability(Enum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    IMPORT_FAILED = "import_failed"
    INIT_FAILED = "init_failed"
    BINARY_MISSING = "binary_missing"


@dataclass(frozen=True)
class AdbDeviceInfo:
    name: str
    adb_path: str
    address: str
    screencap_methods: int
    input_methods: int
    config: dict


class MaaFrameworkDetector:
    _instance = None
    _status: MaaAvailability = MaaAvailability.UNKNOWN
    _status_message: str = ""
    _checked: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def detect(cls) -> MaaAvailability:
        if cls._checked:
            return cls._status

        binary_path = resolve_maafw_path()
        if binary_path and not Path(binary_path).exists():
            cls._status = MaaAvailability.BINARY_MISSING
            cls._status_message = f"MAAFW_BINARY_PATH路径无效: {binary_path}"
            cls._checked = True
            logger.warning(cls._status_message)
            return cls._status

        try:
            from maa.toolkit import Toolkit
        except Exception:
            cls._status = MaaAvailability.IMPORT_FAILED
            cls._status_message = "MAA Framework导入失败，请安装maafw"
            cls._checked = True
            logger.warning(cls._status_message)
            return cls._status

        try:
            Toolkit.init_option(str(Path.cwd()))
            cls._status = MaaAvailability.AVAILABLE
            cls._status_message = "MAA Framework可用"
        except Exception as e:
            cls._status = MaaAvailability.INIT_FAILED
            cls._status_message = f"MAA Framework初始化失败: {e}"
            logger.warning(cls._status_message)

        cls._checked = True
        return cls._status

    @classmethod
    def is_available(cls) -> bool:
        return cls.detect() == MaaAvailability.AVAILABLE

    @classmethod
    def get_status(cls) -> MaaAvailability:
        return cls.detect()

    @classmethod
    def get_status_message(cls) -> str:
        cls.detect()
        return cls._status_message

    @classmethod
    def reset(cls):
        cls._checked = False
        cls._status = MaaAvailability.UNKNOWN
        cls._status_message = ""


class MaaAdbConnector:
    def __init__(self):
        self.screen_width: int = 0
        self.screen_height: int = 0
        self.device_serial: str = ""
        self.is_connected: bool = False
        self._ctrl = None
        self._devices: list[AdbDeviceInfo] = []
        self._selected_device_index: int = -1

    @property
    def devices(self) -> list[AdbDeviceInfo]:
        return self._devices

    @property
    def selected_device(self) -> AdbDeviceInfo | None:
        if 0 <= self._selected_device_index < len(self._devices):
            return self._devices[self._selected_device_index]
        return None

    def _ensure_toolkit(self):
        binary_path = resolve_maafw_path()
        if binary_path:
            os.environ["MAAFW_BINARY_PATH"] = binary_path
        from maa.toolkit import Toolkit
        Toolkit.init_option(str(Path.cwd()))
        return Toolkit

    def find_devices(self, adb_path: str = "") -> list[AdbDeviceInfo]:
        if not MaaFrameworkDetector.is_available():
            logger.warning("MAA Framework不可用，无法发现设备")
            return []

        try:
            Toolkit = self._ensure_toolkit()
            raw_devices = Toolkit.find_adb_devices(specified_adb=adb_path) if adb_path else Toolkit.find_adb_devices()
            seen = {d.address for d in self._devices}
            new_devices = []
            for d in raw_devices:
                info = AdbDeviceInfo(
                    name=d.name,
                    adb_path=str(d.adb_path),
                    address=d.address,
                    screencap_methods=d.screencap_methods,
                    input_methods=d.input_methods,
                    config=d.config if isinstance(d.config, dict) else {},
                )
                if info.address not in seen:
                    new_devices.append(info)
                    self._devices.append(info)
                    seen.add(info.address)
            if new_devices:
                logger.info(f"发现 {len(new_devices)} 个新ADB设备（总计 {len(self._devices)} 个）")
            elif not adb_path:
                logger.info(f"发现 {len(self._devices)} 个ADB设备")
            return self._devices
        except Exception as e:
            logger.error(f"发现ADB设备失败: {e}")
            if not self._devices:
                return []
            return self._devices

    def connect(self, device_index: int = 0) -> bool:
        if not MaaFrameworkDetector.is_available():
            self.is_connected = False
            logger.warning("MAA Framework不可用，无法连接")
            return False

        try:
            binary_path = resolve_maafw_path()
            if binary_path:
                os.environ["MAAFW_BINARY_PATH"] = binary_path

            from maa.toolkit import Toolkit
            from maa.controller import AdbController

            Toolkit.init_option(str(Path.cwd()))

            if not self._devices:
                self.find_devices()

            if not self._devices:
                raise RuntimeError("未找到任何ADB设备，请确保设备已连接且ADB调试已启用")

            if device_index < 0 or device_index >= len(self._devices):
                device_index = 0

            self._selected_device_index = device_index
            device = self._devices[device_index]

            from maa.define import MaaAdbScreencapMethodEnum, MaaAdbInputMethodEnum
            screencap = device.screencap_methods if device.screencap_methods else MaaAdbScreencapMethodEnum.Default
            input_m = device.input_methods if device.input_methods else MaaAdbInputMethodEnum.Default

            self._ctrl = AdbController(
                device.adb_path,
                device.address,
                screencap,
                input_m,
                device.config,
            )

            self._ctrl.post_connection().wait()
            self._ctrl.set_screenshot_use_raw_size(True)

            self._ctrl.post_screencap().wait()
            image = self._ctrl.cached_image
            if image is not None:
                self.screen_height, self.screen_width = image.shape[:2]
            else:
                self.screen_width, self.screen_height = self._get_window_size_fallback()

            self.device_serial = device.address
            self.is_connected = True
            logger.info(f"MAA Framework ADB连接成功: {device.name} ({device.address}), 分辨率: {self.screen_width}x{self.screen_height}")
            return True

        except Exception as e:
            self.is_connected = False
            logger.error(f"MAA Framework ADB连接失败: {e}")
            return False

    def connect_custom(self, address: str, adb_path: str = "") -> bool:
        if not MaaFrameworkDetector.is_available():
            self.is_connected = False
            logger.warning("MAA Framework不可用，无法连接")
            return False

        try:
            Toolkit = self._ensure_toolkit()
            from maa.controller import AdbController
            from maa.define import MaaAdbScreencapMethodEnum, MaaAdbInputMethodEnum

            if not adb_path:
                if self._devices:
                    adb_path = self._devices[0].adb_path
                else:
                    adb_path = str(Path.cwd() / "platform-tools" / "adb.exe")

            self._ctrl = AdbController(
                adb_path,
                address,
                MaaAdbScreencapMethodEnum.Default,
                MaaAdbInputMethodEnum.Default,
                {},
            )

            self._ctrl.post_connection().wait()
            self._ctrl.set_screenshot_use_raw_size(True)

            self._ctrl.post_screencap().wait()
            image = self._ctrl.cached_image
            if image is not None:
                self.screen_height, self.screen_width = image.shape[:2]
            else:
                self.screen_width, self.screen_height = 1920, 1080

            self.device_serial = address
            self.is_connected = True
            self._selected_device_index = -1
            logger.info(f"MAA Framework自定义连接成功: {address}, 分辨率: {self.screen_width}x{self.screen_height}")
            return True

        except Exception as e:
            self.is_connected = False
            logger.error(f"MAA Framework自定义连接失败: {e}")
            return False

    def _get_window_size_fallback(self) -> tuple[int, int]:
        try:
            device = self.selected_device
            if not device:
                return 1920, 1080
            size_cmd = f"{device.adb_path} -s {device.address} shell wm size"
            result = subprocess.run(size_cmd, shell=True, capture_output=True, text=True, check=True, timeout=5)
            output = result.stdout.strip()
            if "Physical size:" in output:
                res_str = output.split("Physical size: ")[1]
            elif "Override size:" in output:
                res_str = output.split("Override size: ")[1]
            else:
                return 1920, 1080
            w, h = map(int, res_str.split("x"))
            return (w, h) if w > h else (h, w)
        except Exception:
            return 1920, 1080

    def capture_screenshot(self) -> np.ndarray | None:
        if not self.is_connected or self._ctrl is None:
            return None
        try:
            self._ctrl.post_screencap().wait()
            image = self._ctrl.cached_image
            if image is not None:
                return image
            logger.error("MAA截图返回None")
            return None
        except Exception as e:
            logger.error(f"MAA截图失败: {e}")
            return None

    def click(self, point: tuple[float, float]):
        if not self.is_connected or self._ctrl is None:
            return
        try:
            x, y = point
            x_coord = int(x * self.screen_width)
            y_coord = int(y * self.screen_height)
            logger.info(f"MAA点击坐标: ({x_coord}, {y_coord})")
            self._ctrl.post_click(x_coord, y_coord).wait()
        except Exception as e:
            logger.error(f"MAA点击失败: {e}")

    def swipe(self, start: tuple[float, float], end: tuple[float, float], duration: int = 500):
        if not self.is_connected or self._ctrl is None:
            return
        try:
            x1, y1 = start
            x2, y2 = end
            x1_coord = int(x1 * self.screen_width)
            y1_coord = int(y1 * self.screen_height)
            x2_coord = int(x2 * self.screen_width)
            y2_coord = int(y2 * self.screen_height)
            logger.info(f"MAA滑动: ({x1_coord},{y1_coord}) -> ({x2_coord},{y2_coord})")
            self._ctrl.post_swipe(x1_coord, y1_coord, x2_coord, y2_coord, duration).wait()
        except Exception as e:
            logger.error(f"MAA滑动失败: {e}")

    def _scan_adb_devices_subprocess(self) -> list[str]:
        try:
            adb_path = ""
            if self._devices:
                adb_path = self._devices[0].adb_path
            if not adb_path or not Path(adb_path).exists():
                adb_path = str(Path.cwd() / "platform-tools" / "adb.exe")
            if not Path(adb_path).exists():
                return []
            result = subprocess.run(
                [adb_path, "devices"],
                capture_output=True, text=True, timeout=5,
            )
            addresses = []
            for line in result.stdout.splitlines():
                parts = line.strip().split("\t")
                if len(parts) == 2 and parts[1] == "device":
                    addresses.append(parts[0])
            return addresses
        except Exception as e:
            logger.debug(f"adb devices扫描失败: {e}")
            return []

    def get_device_list(self) -> list[str]:
        self.find_devices()
        seen = {d.address for d in self._devices}
        extra_addresses = [a for a in self._scan_adb_devices_subprocess() if a not in seen]
        for addr in extra_addresses:
            adb_path = self._devices[0].adb_path if self._devices else str(Path.cwd() / "platform-tools" / "adb.exe")
            self._devices.append(AdbDeviceInfo(
                name="ADB Device",
                adb_path=adb_path,
                address=addr,
                screencap_methods=0,
                input_methods=0,
                config={},
            ))
        if extra_addresses:
            logger.info(f"通过adb devices补充发现 {len(extra_addresses)} 个设备")
        return [f"{d.name} ({d.address})" for d in self._devices]

    def update_device_serial(self, serial: str) -> str:
        self.device_serial = serial
        return serial

    def disconnect(self):
        if self._ctrl is not None:
            try:
                self._ctrl = None
            except Exception:
                pass
        self.is_connected = False

    def stop_adb_server(self):
        adb_path = ""
        device = self.selected_device
        if device:
            adb_path = device.adb_path
        elif self._devices:
            adb_path = self._devices[0].adb_path

        if not adb_path or not Path(adb_path).exists():
            adb_path = str(Path.cwd() / "platform-tools" / "adb.exe")

        if not Path(adb_path).exists():
            logger.warning("未找到adb.exe，跳过关闭ADB server")
            return

        try:
            subprocess.run([adb_path, "kill-server"], capture_output=True, text=True, timeout=5, check=False)
            logger.info("已执行 adb kill-server")
        except Exception as e:
            logger.warning(f"关闭ADB server失败: {e}")
