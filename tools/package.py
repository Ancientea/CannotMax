import subprocess
import shutil
from pathlib import Path
import toml # 导入toml库

# TODO: 需重新适配UV

# 配置区（用户可根据需要修改这些参数）
CONFIG = {
    "venv_dir": ".venv",                  # 虚拟环境目录
    "source_script": "main.py",          # 主程序文件路径
    "icon_file": r"ico/icon_64x64.ico",             # 图标文件路径
    "output_dir": "output",              # 输出目录
    "console": True,
    "add_data": [                        # 需要打包的附加数据
        (r".venv/Lib/site-packages/rapidocr/default_models.yaml", "rapidocr"),
        (r".venv/Lib/site-packages/rapidocr/config.yaml", "rapidocr"),
        # (r".venv/Lib/site-packages/onnxruntime", "onnxruntime"),
    ],
    "copy_files": [                      # 需要复制的额外文件/目录
        r"C:\Windows\System32\msvcp140.dll",
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Windows\System32\vcruntime140_1.dll",
        # "arknights.csv",
        "models/best_model_full.onnx",
        "images",
        "platform-tools",
        "ico",
        "pyproject.toml",
        "monster.csv",
    ]
}

def build_exe():
    """使用PyInstaller打包"""
    venv_python = str(Path(CONFIG["venv_dir"]) / "Scripts" / "python.exe")
    script_path = Path(CONFIG["source_script"])
    icon_path = Path(CONFIG["icon_file"])
    build_cmd = [
        venv_python, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console" if CONFIG["console"] else "--windowed",  # 动态选择参数
        "--name", script_path.stem,
        "--distpath", CONFIG["output_dir"],
        "--workpath", "build",
        "--exclude-module", "train",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        # "--exclude-module", "onnxruntime",
        "--exclude-module", "predict",
        # "--exclude-module", "toml",
        # "--add-binary", ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll;.",
    ]

    if icon_path.exists():
        build_cmd.extend(["--icon", str(icon_path)])

    # 添加附加数据
    for src, dest in CONFIG["add_data"]:
        build_cmd.extend(["--add-data", f"{Path(src).resolve()};{dest}"])

    build_cmd.append(str(script_path))

    try:
        subprocess.check_call(build_cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"打包失败：{e}")
        return False

def copy_additional_files():
    """复制额外文件到输出目录"""
    exe_dir = Path(CONFIG["output_dir"]) / Path(CONFIG["source_script"]).stem
    if not exe_dir.exists():
        print(f"输出目录不存在: {exe_dir}")
        return False

    for item in CONFIG["copy_files"]:
        src = Path(item)
        # 保持相对路径结构
        if src.is_absolute():
            dest = exe_dir / src.name
        else:
            # 对于相对路径，在输出目录中创建相同的目录结构
            dest = exe_dir / src

        if not src.exists():
            print(f"警告：{src} 不存在，跳过复制")
            continue

        try:
            if src.is_dir():
                # 确保目标目录存在
                dest.mkdir(parents=True, exist_ok=True)
                # 特殊处理images目录，排除tmp和nums子目录
                if src.name == "images":
                    def ignore_func(dir, names):
                        """忽略tmp和nums子目录"""
                        dir_path = Path(dir)
                        # 仅在images根目录下排除特定子目录
                        if dir_path.resolve() == src.resolve():
                            return [n for n in names if n in {'tmp', 'nums'}]
                        return []
                    shutil.copytree(src, dest, ignore=ignore_func, dirs_exist_ok=True)
                else:
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                print(f"已复制目录：{src} -> {dest}")
            else:
                # 确保目标文件的父目录存在
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                print(f"已复制文件：{src} -> {dest}")
        except Exception as e:
            print(f"复制失败：{e}")

    return True

def create_zip_archive(project_name, project_version):
    """将输出目录打包为zip文件"""
    output_dir = Path(CONFIG["output_dir"]) / "main"
    if not output_dir.exists():
        print(f"错误：输出目录 '{output_dir}' 不存在，无法创建zip文件。")

    zip_name = Path(CONFIG["output_dir"]) / f"{project_name}-{project_version}"
    try:
        # shutil.make_archive 会自动添加 .zip 扩展名
        shutil.make_archive(zip_name, 'zip', root_dir=output_dir)
        print(f"已创建zip文件：{zip_name}.zip")
    except Exception as e:
        print(f"创建zip文件失败：{e}")

def main():
    with open("pyproject.toml", "r", encoding="utf-8") as f:
        pyproject_data = toml.load(f)
    project_name = pyproject_data["project"]["name"]
    project_version = pyproject_data["project"]["version"]

    if not build_exe():
        return

    if not copy_additional_files():
        return

    print(f"\n打包成功！输出目录：{Path(CONFIG['output_dir']).resolve()}")

    create_zip_archive(project_name, project_version)

if __name__ == "__main__":
    main()