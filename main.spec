# -*- mode: python ; coding: utf-8 -*-
import os

# 定义项目根目录（spec文件所在目录）
block_cipher = None
project_root = os.path.abspath(os.getcwd())

a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        ('.venv/Lib/site-packages/rapidocr/default_models.yaml', 'rapidocr'),
        ('.venv/Lib/site-packages/rapidocr/config.yaml', 'rapidocr'),
        ('.venv/Lib/site-packages/rapidocr/models', 'rapidocr/models')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['train', 'torch', 'torchvision', 'matplotlib', 'sklearn', 'scikit-learn', 'scipy', 'PyQt6.QtPdf', 'PyQt6.QtNetwork', 'predict', 'onnxscript'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 【核心：自动过滤不需要的二进制文件】
# 这会过滤掉 a.binaries 中包含指定名称的所有 DLL
unwanted_bins = ['Qt6Pdf', 'Qt6Network', 'opengl32sw', 'opencv_videoio_ffmpeg']
a.binaries = [x for x in a.binaries if not any(bad in x[0] for bad in unwanted_bins)]


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['ico\\icon_64x64.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
