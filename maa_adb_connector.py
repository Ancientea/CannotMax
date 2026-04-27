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
class ConnectionType:
    type_id: str
    display_name: str
    default_address: str
    description: str


@dataclass(frozen=True)
class InputMethodOption:
    method_id: str
    enum_value: int
    display_name: str
    description: str


@dataclass(frozen=True)
class MaaConnectionConfig:
    maa_binary_path: str = ""
    adb_path: str = r".\platform-tools\adb.exe"
    device_serial: str = ""
    screencap_method: int = 1
    input_method: int = 4
    screenshot_use_raw_size: bool = True
    config: dict = field(default_factory=dict)


@dataclass
class AdapterState:
    use_maa: bool = False
    maa_availability: MaaAvailability = MaaAvailability.UNKNOWN
    active_connector_type: str = "legacy"
    connection_type_id: str = "adb"
    input_method_id: str = "maatouch"


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

        try:
            from maa.toolkit import Toolkit
        except ImportError:
            cls._status = MaaAvailability.IMPORT_FAILED
            cls._status_message = "MAA Framework导入失败，请安装maa.library"
            cls._checked = True
            logger.warning(cls._status_message)
            return cls._status

        binary_path = resolve_maafw_path()
        if binary_path and not Path(binary_path).exists():
            cls._status = MaaAvailability.BINARY_MISSING
            cls._status_message = f"MAAFW_BINARY_PATH路径无效: {binary_path}"
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


class ConnectionTypeRegistry:
    _types: list[ConnectionType] = [
        ConnectionType("adb", "ADB连接", "", "通用ADB连接，需手动指定设备地址"),
        ConnectionType("ldplayer", "雷电模拟器", "emulator-5554", "雷电模拟器默认ADB地址"),
        ConnectionType("mumu", "MuMu模拟器", "127.0.0.1:7555", "MuMu模拟器默认ADB地址"),
        ConnectionType("mumu12", "MuMu12模拟器", "127.0.0.1:16384", "MuMu12模拟器默认ADB地址"),
        ConnectionType("bluestacks", "蓝叠模拟器", "127.0.0.1:5555", "蓝叠模拟器默认ADB地址"),
        ConnectionType("nox", "夜神模拟器", "127.0.0.1:62001", "夜神模拟器默认ADB地址"),
    ]

    @classmethod
    def get_all_types(cls) -> list[ConnectionType]:
        return cls._types

    @classmethod
    def get_default_address(cls, type_id: str) -> str:
        for ct in cls._types:
            if ct.type_id == type_id:
                return ct.default_address
        return ""

    @classmethod
    def get_type_by_id(cls, type_id: str) -> ConnectionType | None:
        for ct in cls._types:
            if ct.type_id == type_id:
                return ct
        return None


class InputMethodRegistry:
    _methods: list[InputMethodOption] = [
        InputMethodOption("adb_shell", 1, "AdbShell", "ADB shell input命令，兼容性最高"),
        InputMethodOption("minitouch_adb_key", 2, "MinitouchAndAdbKey", "minitouch注入+ADB按键，低延迟需root"),
        InputMethodOption("maatouch", 4, "Maatouch", "Maatouch注入，低延迟MAA自带"),
        InputMethodOption("emulator_extras", 8, "EmulatorExtras", "模拟器扩展接口，仅特定模拟器支持"),
    ]

    @classmethod
    def get_all_methods(cls) -> list[InputMethodOption]:
        return cls._methods

    @classmethod
    def get_method_by_id(cls, method_id: str) -> InputMethodOption | None:
        for m in cls._methods:
            if m.method_id == method_id:
                return m
        return None

    @classmethod
    def get_enum_value_by_id(cls, method_id: str) -> int:
        m = cls.get_method_by_id(method_id)
        return m.enum_value if m else 4

    @classmethod
    def get_default_method(cls) -> InputMethodOption:
        return cls.get_method_by_id("maatouch")


class MaaAdbConnector:
    def __init__(self, config: MaaConnectionConfig | None = None):
        self._config = config or MaaConnectionConfig()
        self.screen_width: int = 0
        self.screen_height: int = 0
        self.device_serial: str = self._config.device_serial
        self.is_connected: bool = False
        self.is_maa_available: bool = False
        self._ctrl = None
        self._connection_type: str = "adb"

    def connect(self):
        if not MaaFrameworkDetector.is_available():
            self.is_maa_available = False
            logger.warning("MAA Framework不可用，MaaAdbConnector无法连接")
            self.is_connected = False
            return

        try:
            binary_path = self._config.maa_binary_path or resolve_maafw_path()
            if binary_path:
                os.environ["MAAFW_BINARY_PATH"] = binary_path

            from maa.toolkit import Toolkit
            from maa.controller import AdbController

            Toolkit.init_option(str(Path.cwd()))

            target_serial = self.device_serial if self.device_serial else "127.0.0.1:5555"
            self.device_serial = target_serial

            adb_path = str(Path(self._config.adb_path).resolve())
            self._ctrl = AdbController(
                adb_path,
                target_serial,
                screencap_methods=self._config.screencap_method,
                input_methods=self._config.input_method,
                config=self._config.config,
            )

            self._ctrl.post_connection().wait()
            self._ctrl.set_screenshot_use_raw_size(self._config.screenshot_use_raw_size)

            self._ctrl.post_screencap().wait()
            image = self._ctrl.cached_image
            if image is not None:
                self.screen_height, self.screen_width = image.shape[:2]
            else:
                self.screen_width, self.screen_height = self._get_window_size_fallback()

            self.is_maa_available = True
            self.is_connected = True
            logger.info(f"MAA Framework ADB连接成功: {target_serial}, 分辨率: {self.screen_width}x{self.screen_height}")

        except Exception as e:
            self.is_maa_available = False
            self.is_connected = False
            logger.error(f"MAA Framework ADB连接失败: {e}")

    def _get_window_size_fallback(self) -> tuple[int, int]:
        try:
            size_cmd = f"{self._config.adb_path} -s {self.device_serial} shell wm size"
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

    def get_device_list(self) -> list[str]:
        try:
            from maa.toolkit import AdbDevice
            devices = AdbDevice.find()
            if devices:
                return [d.name for d in devices]
        except Exception:
            logger.debug("MAA AdbDevice.find()失败，降级到subprocess")
        try:
            device_cmd = f"{self._config.adb_path} devices"
            result = subprocess.run(device_cmd, shell=True, capture_output=True, text=True, timeout=5)
            devices = []
            for line in result.stdout.split("\n"):
                if "\tdevice" in line:
                    devices.append(line.split("\t")[0])
            return devices
        except Exception as e:
            logger.error(f"获取设备列表失败: {e}")
            return []

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
        self.is_maa_available = False

    def set_config(self, config: MaaConnectionConfig):
        self._config = config
        if config.device_serial:
            self.device_serial = config.device_serial


class AdbConnectorAdapter:
    def __init__(self, adb_path: str = r".\platform-tools\adb.exe"):
        import loadData
        self._legacy_connector = loadData.AdbConnector()
        self._maa_connector: MaaAdbConnector | None = None
        self._use_maa: bool = False
        self._maa_config = MaaConnectionConfig(adb_path=adb_path)
        self._state = AdapterState()

    @property
    def is_connected(self) -> bool:
        if self._use_maa and self._maa_connector:
            return self._maa_connector.is_connected
        return self._legacy_connector.is_connected

    @property
    def screen_width(self) -> int:
        if self._use_maa and self._maa_connector:
            return self._maa_connector.screen_width
        return self._legacy_connector.screen_width

    @property
    def screen_height(self) -> int:
        if self._use_maa and self._maa_connector:
            return self._maa_connector.screen_height
        return self._legacy_connector.screen_height

    @property
    def device_serial(self) -> str:
        if self._use_maa and self._maa_connector:
            return self._maa_connector.device_serial
        return self._legacy_connector.device_serial

    @property
    def is_maa_available(self) -> bool:
        return self._use_maa

    @property
    def active_implementation(self) -> str:
        return "maa" if self._use_maa else "legacy"

    @property
    def state(self) -> AdapterState:
        return self._state

    def get_config(self) -> MaaConnectionConfig:
        return self._maa_config

    def set_maa_binary_path(self, path: str):
        self._maa_config = MaaConnectionConfig(
            maa_binary_path=path,
            adb_path=self._maa_config.adb_path,
            device_serial=self._maa_config.device_serial,
            screencap_method=self._maa_config.screencap_method,
            input_method=self._maa_config.input_method,
            screenshot_use_raw_size=self._maa_config.screenshot_use_raw_size,
            config=self._maa_config.config,
        )
        if path:
            os.environ["MAAFW_BINARY_PATH"] = path
        MaaFrameworkDetector.reset()

    def set_connection_type(self, type_id: str):
        self._state.connection_type_id = type_id
        default_addr = ConnectionTypeRegistry.get_default_address(type_id)
        if default_addr:
            self._maa_config = MaaConnectionConfig(
                maa_binary_path=self._maa_config.maa_binary_path,
                adb_path=self._maa_config.adb_path,
                device_serial=default_addr,
                screencap_method=self._maa_config.screencap_method,
                input_method=self._maa_config.input_method,
                screenshot_use_raw_size=self._maa_config.screenshot_use_raw_size,
                config=self._maa_config.config,
            )
            self._legacy_connector.device_serial = default_addr

    def set_input_method(self, method_id: str):
        enum_value = InputMethodRegistry.get_enum_value_by_id(method_id)
        self._state.input_method_id = method_id
        self._maa_config = MaaConnectionConfig(
            maa_binary_path=self._maa_config.maa_binary_path,
            adb_path=self._maa_config.adb_path,
            device_serial=self._maa_config.device_serial,
            screencap_method=self._maa_config.screencap_method,
            input_method=enum_value,
            screenshot_use_raw_size=self._maa_config.screenshot_use_raw_size,
            config=self._maa_config.config,
        )

    def set_device_serial(self, serial: str):
        self._maa_config = MaaConnectionConfig(
            maa_binary_path=self._maa_config.maa_binary_path,
            adb_path=self._maa_config.adb_path,
            device_serial=serial,
            screencap_method=self._maa_config.screencap_method,
            input_method=self._maa_config.input_method,
            screenshot_use_raw_size=self._maa_config.screenshot_use_raw_size,
            config=self._maa_config.config,
        )
        self._legacy_connector.device_serial = serial

    def connect(self):
        if MaaFrameworkDetector.is_available():
            try:
                maa_connector = MaaAdbConnector(self._maa_config)
                maa_connector.device_serial = self._legacy_connector.device_serial or self._maa_config.device_serial
                maa_connector.connect()
                if maa_connector.is_connected:
                    self._maa_connector = maa_connector
                    self._use_maa = True
                    self._state.use_maa = True
                    self._state.maa_availability = MaaAvailability.AVAILABLE
                    self._state.active_connector_type = "maa"
                    logger.info("AdbConnectorAdapter: 使用MAA Framework实现")
                    return
            except Exception as e:
                logger.warning(f"MAA Framework连接失败，降级到自有实现: {e}")

        self._use_maa = False
        self._state.use_maa = False
        self._state.maa_availability = MaaFrameworkDetector.get_status()
        self._state.active_connector_type = "legacy"
        logger.info("AdbConnectorAdapter: 降级到自有ADB实现")
        self._legacy_connector.connect()

    def capture_screenshot(self) -> np.ndarray | None:
        if self._use_maa and self._maa_connector:
            return self._maa_connector.capture_screenshot()
        return self._legacy_connector.capture_screenshot()

    def click(self, point: tuple[float, float]):
        if self._use_maa and self._maa_connector:
            self._maa_connector.click(point)
        else:
            self._legacy_connector.click(point)

    def swipe(self, start: tuple[float, float], end: tuple[float, float], duration: int = 500):
        if self._use_maa and self._maa_connector:
            self._maa_connector.swipe(start, end, duration)
        else:
            logger.warning("自有ADB实现不支持滑动操作")

    def get_device_list(self) -> list[str]:
        if self._use_maa and self._maa_connector:
            return self._maa_connector.get_device_list()
        return self._legacy_connector.get_device_list()

    def update_device_serial(self, serial: str) -> str:
        self.set_device_serial(serial)
        if self._use_maa and self._maa_connector:
            return self._maa_connector.update_device_serial(serial)
        return self._legacy_connector.update_device_serial(serial)

    def disconnect(self):
        if self._maa_connector:
            self._maa_connector.disconnect()
            self._maa_connector = None
        self._use_maa = False
        self._state.use_maa = False
        self._state.active_connector_type = "legacy"
