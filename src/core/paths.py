import sys
from pathlib import Path


if getattr(sys, 'frozen', False):
    _INTERNAL_DIR = Path(sys._MEIPASS)
    EXE_DIR = _INTERNAL_DIR.parent
    PROJECT_ROOT = EXE_DIR
    SRC_DIR = _INTERNAL_DIR / "src"
else:
    _INTERNAL_DIR = None
    EXE_DIR = Path(__file__).resolve().parents[2]
    PROJECT_ROOT = EXE_DIR
    SRC_DIR = PROJECT_ROOT / "src"

RESOURCES_DIR = SRC_DIR / "resources"
ASSETS_DIR = RESOURCES_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
PROCESS_IMAGES_DIR = IMAGES_DIR / "process"
TMP_IMAGES_DIR = IMAGES_DIR / "tmp"
DATA_DIR = RESOURCES_DIR / "data"
SIMULATION_DIR = SRC_DIR / "simulation"
TOOLS_DIR = SRC_DIR / "tools"
VENDOR_DIR = PROJECT_ROOT / "vendor"
MODELS_DIR = PROJECT_ROOT / "models"


def resource_path(*parts: str) -> Path:
    return RESOURCES_DIR.joinpath(*parts)


def image_path(name: str) -> Path:
    return IMAGES_DIR / f"{name}.png"


def process_image_path(name: str | int) -> Path:
    return PROCESS_IMAGES_DIR / f"{name}.png"


def data_path(name: str) -> Path:
    return DATA_DIR / name


def simulation_path(name: str) -> Path:
    return SIMULATION_DIR / name


def ensure_tmp_images_dir() -> Path:
    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_IMAGES_DIR
