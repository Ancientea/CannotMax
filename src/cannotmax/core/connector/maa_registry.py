"""MAA Framework registry and detection utilities.

Provides:
- MaaFrameworkDetector: Singleton to check MAA Framework availability
- ConnectionTypeRegistry: Predefined emulator addresses (LDPlayer, MuMu, etc.)
- InputMethodRegistry: Input method options (maatouch, adb_shell, etc.)
- MaaAvailability: Enum for MAA availability states
"""
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class MaaAvailability(Enum):
    """MAA Framework availability states."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    IMPORT_FAILED = "import_failed"
    INIT_FAILED = "init_failed"


@dataclass(frozen=True)
class ConnectionType:
    """Emulator connection type definition."""
    type_id: str
    display_name: str
    default_address: str
    description: str


@dataclass(frozen=True)
class InputMethodOption:
    """Input method option definition."""
    method_id: str
    enum_value: int
    display_name: str
    description: str


class MaaFrameworkDetector:
    """Singleton detector for MAA Framework availability.
    
    Usage:
        if MaaFrameworkDetector.is_available():
            # Use MAA Framework
        else:
            # Use legacy ADB/Win32
    """
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
        """Detect MAA Framework availability."""
        if cls._checked:
            return cls._status

        try:
            from maa.toolkit import Toolkit
        except Exception:
            cls._status = MaaAvailability.IMPORT_FAILED
            cls._status_message = "MAA Framework 导入失败，请安装 maa.library"
            cls._checked = True
            logger.warning(cls._status_message)
            return cls._status

        try:
            logger.debug("尝试初始化 MAA Toolkit")
            Toolkit.init_option(str(Path.cwd()))
            logger.debug("MAA Toolkit 初始化成功")
            cls._status = MaaAvailability.AVAILABLE
            logger.debug("MAA Framework 可用")
            cls._status_message = "MAA Framework 可用"
            logger.debug(cls._status_message)
        except Exception as e:
            cls._status = MaaAvailability.INIT_FAILED
            cls._status_message = f"MAA Framework 初始化失败：{e}"
            logger.warning(cls._status_message)

        cls._checked = True
        return cls._status

    @classmethod
    def is_available(cls) -> bool:
        """Check if MAA Framework is available."""
        return cls.detect() == MaaAvailability.AVAILABLE

    @classmethod
    def get_status(cls) -> MaaAvailability:
        """Get current MAA availability status."""
        return cls.detect()

    @classmethod
    def get_status_message(cls) -> str:
        """Get human-readable status message."""
        cls.detect()
        return cls._status_message

    @classmethod
    def reset(cls):
        """Reset detection state for re-detection."""
        cls._checked = False
        cls._status = MaaAvailability.UNKNOWN
        cls._status_message = ""


class ConnectionTypeRegistry:
    """Registry of emulator connection types with default addresses."""
    
    _types: list[ConnectionType] = [
        ConnectionType("adb", "ADB 连接", "", "通用 ADB 连接，需手动指定设备地址"),
        ConnectionType("ldplayer", "雷电模拟器", "emulator-5554", "雷电模拟器默认 ADB 地址"),
        ConnectionType("mumu", "MuMu 模拟器", "127.0.0.1:7555", "MuMu 模拟器默认 ADB 地址"),
        ConnectionType("mumu12", "MuMu12 模拟器", "127.0.0.1:16384", "MuMu12 模拟器默认 ADB 地址"),
        ConnectionType("bluestacks", "蓝叠模拟器", "127.0.0.1:5555", "蓝叠模拟器默认 ADB 地址"),
        ConnectionType("nox", "夜神模拟器", "127.0.0.1:62001", "夜神模拟器默认 ADB 地址"),
    ]

    @classmethod
    def get_all_types(cls) -> list[ConnectionType]:
        """Get all connection types."""
        return cls._types

    @classmethod
    def get_default_address(cls, type_id: str) -> str:
        """Get default address for connection type."""
        for ct in cls._types:
            if ct.type_id == type_id:
                return ct.default_address
        return ""

    @classmethod
    def get_type_by_id(cls, type_id: str) -> ConnectionType | None:
        """Get connection type by ID."""
        for ct in cls._types:
            if ct.type_id == type_id:
                return ct
        return None


class InputMethodRegistry:
    """Registry of input methods for MAA Framework."""
    
    _methods: list[InputMethodOption] = [
        InputMethodOption("adb_shell", 1, "AdbShell", "ADB shell input 命令，兼容性最高"),
        InputMethodOption("minitouch_adb_key", 2, "MinitouchAndAdbKey", "minitouch 注入+ADB 按键，低延迟需 root"),
        InputMethodOption("maatouch", 4, "Maatouch", "Maatouch 注入，低延迟 MAA 自带"),
        InputMethodOption("emulator_extras", 8, "EmulatorExtras", "模拟器扩展接口，仅特定模拟器支持"),
    ]

    @classmethod
    def get_all_methods(cls) -> list[InputMethodOption]:
        """Get all input methods."""
        return cls._methods

    @classmethod
    def get_method_by_id(cls, method_id: str) -> InputMethodOption | None:
        """Get input method by ID."""
        for m in cls._methods:
            if m.method_id == method_id:
                return m
        return None

    @classmethod
    def get_enum_value_by_id(cls, method_id: str) -> int:
        """Get enum value for MAA Framework by method ID."""
        m = cls.get_method_by_id(method_id)
        return m.enum_value if m else 4

    @classmethod
    def get_default_method(cls) -> InputMethodOption:
        """Get default input method (maatouch)."""
        return cls.get_method_by_id("maatouch")


__all__ = [
    "MaaAvailability",
    "ConnectionType",
    "InputMethodOption",
    "MaaFrameworkDetector",
    "ConnectionTypeRegistry",
    "InputMethodRegistry",
]
