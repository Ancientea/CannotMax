from pathlib import Path
import zipfile
import re
from datetime import datetime


def create_zip_package(output_zip_path):
    # 定义文件和文件夹路径
    data_folder = Path("data")

    # 获取所有符合日期格式的目录
    date_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$")
    time_folders = [
        folder
        for folder in data_folder.iterdir()
        if folder.is_dir() and date_pattern.match(folder.name)
    ]
    if not time_folders:
        print("未找到输出目录！")
        return

    # 创建压缩包
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 添加所有符合日期格式的目录及其内容
        for folder in time_folders:
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    # 在压缩包中保留相对目录结构
                    arcname = file_path.relative_to(data_folder)
                    zipf.write(file_path, arcname=str(arcname))

    print(f"压缩包已创建：{output_zip_path}")


# 使用当前时间生成输出文件名
current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
output_zip = f"arknights_package_{current_time}.zip"

# 调用函数创建压缩包
create_zip_package(output_zip)
