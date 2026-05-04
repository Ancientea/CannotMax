import shutil
import subprocess
import sys
from pathlib import Path

import toml


def _configure_utf8_stdio():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_stdio()

CONFIG = {
    "venv_dir": ".venv",
    "output_dir": "output",
    "specs": ["cannotmax.spec"],
    "copy_files": [
        "images",
        "3rdparty/platform-tools",
        "ico",
        "pyproject.toml",
        "config/battlefield_recognize",
        "monster.csv",
        "monster_greenvine.csv",
        "多开管理器.bat",
    ],
}


def _venv_python():
    return str(Path(CONFIG["venv_dir"]) / "Scripts" / "python.exe")


def build_specs():
    for spec in CONFIG["specs"]:
        print(f"\n{'=' * 40}")
        print(f"Building {spec} ...")
        cmd = [
            _venv_python(),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            CONFIG["output_dir"],
            "--workpath",
            "build",
            spec,
        ]
        try:
            subprocess.check_call(cmd)
            print(f"{spec} 打包完成")
        except subprocess.CalledProcessError as e:
            print(f"{spec} 打包失败：{e}")
            return False
    return True


def _ignore_images(dir, names):
    if Path(dir).resolve() == Path("images").resolve():
        return [n for n in names if n in {"tmp", "nums"}]
    return []


def copy_additional_files():
    output = Path(CONFIG["output_dir"])
    exe_dir = output / "cannotmax"
    if not exe_dir.exists():
        print(f"警告：输出目录 {exe_dir} 不存在，跳过文件复制")
        return True

    for item in CONFIG["copy_files"]:
        src = Path(item)
        if src.is_absolute():
            dest = exe_dir / src.name
        else:
            dest = exe_dir / src

        if not src.exists():
            print(f"  警告：{src} 不存在，跳过")
            continue

        try:
            if src.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                if src.name == "images":
                    shutil.copytree(
                        src, dest, ignore=_ignore_images, dirs_exist_ok=True
                    )
                else:
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                print(f"  目录 {src} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                print(f"  文件 {src} -> {dest}")
        except Exception as e:
            print(f"  复制失败：{e}")
    return True


def create_zip_archive(project_name, project_version):
    output = Path(CONFIG["output_dir"])
    zip_root = output / f"{project_name}-{project_version}"
    try:
        shutil.make_archive(str(zip_root), "zip", root_dir=output, base_dir="cannotmax")
        print(f"已创建 zip：{zip_root}.zip")
    except Exception as e:
        print(f"创建 zip 失败：{e}")


def main():
    with open("pyproject.toml", "r", encoding="utf-8") as f:
        pyproject_data = toml.load(f)
    project_name = pyproject_data["project"]["name"]
    project_version = pyproject_data["project"]["version"]

    print(f"开始打包 {project_name} v{project_version} ...")

    if not build_specs():
        return

    if not copy_additional_files():
        return

    print(f"\n打包成功！输出目录：{Path(CONFIG['output_dir']).resolve()}")

    create_zip_archive(project_name, project_version)


if __name__ == "__main__":
    main()
