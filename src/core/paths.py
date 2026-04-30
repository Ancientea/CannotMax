from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
RESOURCES_DIR = SRC_DIR / "resources"
ASSETS_DIR = RESOURCES_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
PROCESS_IMAGES_DIR = IMAGES_DIR / "process"
TMP_IMAGES_DIR = IMAGES_DIR / "tmp"
DATA_DIR = RESOURCES_DIR / "data"
SIMULATION_DIR = SRC_DIR / "simulation"
TOOLS_DIR = SRC_DIR / "tools"


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
