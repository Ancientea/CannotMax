"""
Path configuration for CannotMax.

集中管理项目路径，使用 pathlib.Path 对象。所有文件路径应通过此模块引用，禁止在代码中硬编码路径字符串。
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 数据相关路径
DATA_DIR = PROJECT_ROOT / "data"
COMPRESSED_DIR = DATA_DIR / "compressed"
IMAGES_DATA_DIR = DATA_DIR / "images"
ARKNIGHTS_DATA_CSV = DATA_DIR / "arknights.csv"

# 图片资源路径
IMAGES_DIR = PROJECT_ROOT / "images"
MONSTER_IMAGES_DIR = IMAGES_DIR / "monsters"
PROCESS_IMAGES_DIR = IMAGES_DIR / "process"
LOGIN_IMAGES_DIR = IMAGES_DIR / "login"
SAMPLES_IMAGES_DIR = IMAGES_DIR / "samples"
TMP_IMAGES_DIR = IMAGES_DIR / "tmp"

# 图标路径
ICO_DIR = PROJECT_ROOT / "ico"

# 模型相关路径
MODELS_DIR = PROJECT_ROOT / "models"

# 配置相关路径
CONFIG_DIR = PROJECT_ROOT / "config"
BATTLEFIELD_RECOGNIZE_DIR = CONFIG_DIR / "battlefield_recognize"

# 第三方工具路径
THIRDPARTY_DIR = PROJECT_ROOT / "3rdparty"
ADB_PATH = THIRDPARTY_DIR / "platform-tools" / "adb.exe"

# 构建输出
OUTPUT_DIR = PROJECT_ROOT / "output"

# 多开端口配置
MULTI_PORTS_FILE = CONFIG_DIR / "multi_ports.txt"

# 数据文件
MONSTER_GREENVINE_CSV = PROJECT_ROOT / "monster_greenvine.csv"
MONSTER_CSV = PROJECT_ROOT / "monster.csv"

# 导出
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "COMPRESSED_DIR",
    "ARKNIGHTS_DATA_CSV",
    "IMAGES_DIR",
    "MONSTER_IMAGES_DIR",
    "PROCESS_IMAGES_DIR",
    "LOGIN_IMAGES_DIR",
    "SAMPLES_IMAGES_DIR",
    "TMP_IMAGES_DIR",
    "ICO_DIR",
    "MODELS_DIR",
    "CONFIG_DIR",
    "BATTLEFIELD_RECOGNIZE_DIR",
    "THIRDPARTY_DIR",
    "ADB_PATH",
    "OUTPUT_DIR",
    "MULTI_PORTS_FILE",
    "MONSTER_GREENVINE_CSV",
    "MONSTER_CSV",
]
