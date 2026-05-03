"""
Path configuration for CannotMax.

集中管理项目路径，使用 pathlib.Path 对象。

规则：
1. 项目中不得出现硬编码的路径字符串，所有路径必须从本模块获取。
2. 变量命名规范：目录以 DIR 结尾，文件以文件格式结尾（如 CSV、JSON、PNG、ONNX）。

所有路径应通过此模块引用，禁止在代码中硬编码路径字符串。
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# 数据相关路径
DATA_DIR = PROJECT_ROOT_DIR / "data"
COMPRESSED_DIR = DATA_DIR / "compressed"
IMAGES_DATA_DIR = DATA_DIR / "images"
ARKNIGHTS_DATA_CSV = DATA_DIR / "arknights.csv"
ARKNIGHTS_DATA_CSV_CLEANED = DATA_DIR / "arknights_cleaned.csv"
ARKNIGHTS_DATA_CSV_WITH_FIELD_RECOGNIZE = (
    DATA_DIR / "arknights_with_field_recognize_v2.csv"
)

# 图片资源路径
IMAGES_DIR = PROJECT_ROOT_DIR / "images"
MONSTER_IMAGES_DIR = IMAGES_DIR / "monsters"
PROCESS_IMAGES_DIR = IMAGES_DIR / "process"
LOGIN_IMAGES_DIR = IMAGES_DIR / "login"
SAMPLES_IMAGES_DIR = IMAGES_DIR / "samples"
TMP_IMAGES_DIR = IMAGES_DIR / "tmp"

# 图标路径
ICO_DIR = PROJECT_ROOT_DIR / "ico"

# 模型相关路径
MODELS_DIR = PROJECT_ROOT_DIR / "models"
DEFAULT_PREDICTOR_PTH = MODELS_DIR / "predictor" / "best_model_full.pth"
DEFAULT_PREDICTOR_ONNX = MODELS_DIR / "predictor" / "best_model_full.onnx"
DEFAULT_FIELD_RECOGNIZER_PTH = MODELS_DIR / "battlefield_recognizer"
DEFAULT_FIELD_RECOGNIZER_ONNX = (
    MODELS_DIR / "battlefield_recognizer" / "field_recognize.onnx"
)

# 配置相关路径
CONFIG_DIR = PROJECT_ROOT_DIR / "config"
APP_CONFIG_JSON = CONFIG_DIR / "app.json"
FIELD_RECOGNITION_CLASS2IDX_JSON = (
    CONFIG_DIR / "battlefield_recognize" / "class_to_idx.json"
)

# 第三方工具路径
THIRDPARTY_DIR = PROJECT_ROOT_DIR / "3rdparty"
ADB_EXE = THIRDPARTY_DIR / "platform-tools" / "adb.exe"

# 构建输出
OUTPUT_DIR = PROJECT_ROOT_DIR / "output"
PACKAGE_OUTPUT_DIR = OUTPUT_DIR / "data"
PACKAGE_FORMAT = "arknights_package_%Y%m%d_%H%M%S.zip"

# 多开端口配置
MULTI_PORTS_CONFIG_TXT = CONFIG_DIR / "multi_ports.txt"

# 数据文件
MONSTER_CSV = PROJECT_ROOT_DIR / "monster_greenvine.csv"

# 导出
__all__ = [
    "PROJECT_ROOT_DIR",
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
    "THIRDPARTY_DIR",
    "ADB_EXE",
    "OUTPUT_DIR",
    "MULTI_PORTS_CONFIG_TXT",
    "MONSTER_CSV",
    "PACKAGE_OUTPUT_DIR",
    "PACKAGE_FORMAT",
]
