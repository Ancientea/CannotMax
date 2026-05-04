import re
import shutil
import zipfile
from datetime import datetime

from cannotmax.config.paths import DATA_DIR, PACKAGE_FORMAT, PACKAGE_OUTPUT_DIR


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

    output_dir = output_zip_path.parent
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


def package_data(format: str = PACKAGE_FORMAT) -> str | None:
    output_dir = PACKAGE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / datetime.now().strftime(format)

    success = create_zip_package(output_zip)
    if not success:
        return None
    return str(output_zip)


if __name__ == "__main__":
    package_data()
