import zipfile
from pathlib import Path
import shutil
import os

def unpackage():
    current_dir = Path(__file__).parent
    extract_dir = current_dir / "tmp"
    package_dir = current_dir / "package"
    
    # 1. 清空 tmp 目录
    if extract_dir.exists():
        print(f"正在清空目录: {extract_dir}")
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if not package_dir.exists():
        print(f"未找到目录: {package_dir}")
        return

    zip_files = list(package_dir.glob("*.zip"))
    print(f"找到 {len(zip_files)} 个压缩包，准备解压...")

    for zip_path in zip_files:
        print(f"正在解压: {zip_path.name}...")
        # 解压到临时目录以处理命名冲突
        temp_dir = extract_dir / f"_temp_{zip_path.stem}"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 将内容移入 tmp 目录，并处理冲突
        for item in temp_dir.iterdir():
            target_name = item.name
            counter = 1
            while (extract_dir / target_name).exists():
                target_name = f"{item.name}__{counter}"
                counter += 1
            
            if target_name != item.name:
                print(f"提示：'{item.name}' 已存在，重命名为 '{target_name}'")
            shutil.move(str(item), str(extract_dir / target_name))
        
        shutil.rmtree(temp_dir)
    print("解压完成。")

if __name__ == "__main__":
    unpackage()
