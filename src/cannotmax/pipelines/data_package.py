import re
import shutil
import zipfile
from datetime import datetime

from cannotmax.config.paths import DATA_DIR
from cannotmax.config.settings import get_data_package_output_dir


def create_zip_package(output_zip_path):
    data_folder = DATA_DIR

    date_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$")
    time_folders = [
        folder
        for folder in data_folder.iterdir()
        if folder.is_dir() and date_pattern.match(folder.name)
    ]
    if not time_folders:
        return False

    output_dir = get_data_package_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in time_folders:
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(data_folder)
                    zipf.write(file_path, arcname=str(arcname))

    for folder in time_folders:
        try:
            shutil.rmtree(folder)
            print(f"已删除文件夹：{folder}")
        except Exception as e:
            print(f"删除文件夹 {folder} 时出错：{e}")

    print(f"压缩包已创建：{output_zip_path}")
    return True


def package_data():
    output_dir = get_data_package_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_zip = output_dir / f"arknights_package_{current_time}.zip"

    success = create_zip_package(str(output_zip))
    if not success:
        return None
    return str(output_zip)


if __name__ == "__main__":
    package_data()
