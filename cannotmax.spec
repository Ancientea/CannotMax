# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None
project_root = os.path.abspath(os.getcwd())

# 主程序分析
a_main = Analysis(
    ['src/cannotmax/console.py', 'src/cannotmax/_multi.py'],
    pathex=[project_root, os.path.join(project_root, 'src')],
    binaries=collect_dynamic_libs('maa'),
    datas=[
        ('.venv/Lib/site-packages/rapidocr/default_models.yaml', 'rapidocr'),
        ('.venv/Lib/site-packages/rapidocr/config.yaml', 'rapidocr'),
        ('.venv/Lib/site-packages/rapidocr/models', 'rapidocr/models'),
    ] + collect_data_files('maa'),
    hiddenimports=['maa', 'maa.controller', 'maa.toolkit', 'maa.resource', 'maa.library',
                   'cannotsim', 'cannotsim.main_sim', 'cannotsim.battle_field',
                   'cannotsim.monsters', 'cannotsim.unit', 'cannotsim.utils', 'cannotsim.vector2d'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'cannotdl.training', 'cannotdl.core',  # PyTorch-dependent
        'torch', 'torchvision', 'matplotlib',
        'sklearn', 'scikit-learn', 'scipy',
        'PyQt6.QtPdf', 'PyQt6.QtNetwork',
        'onnxscript',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

unwanted_bins = ['Qt6Pdf', 'Qt6Network', 'opengl32sw', 'opencv_videoio_ffmpeg']
a_main.binaries = [x for x in a_main.binaries if not any(bad in x[0] for bad in unwanted_bins)]

pyz_main = PYZ(a_main.pure, a_main.zipped_data, cipher=block_cipher)

exe_main = EXE(
    pyz_main,
    a_main.scripts,
    [],
    exclude_binaries=True,
    name='cannotmax',
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

# 多开管理器（薄封装，使用 _multi.py 作为入口）
_multi_scripts = [s for s in a_main.scripts if '_multi' in s[0]] + \
                [s for s in a_main.scripts if '_multi' not in s[0]]

exe_multi = EXE(
    pyz_main,
    _multi_scripts,
    [],
    exclude_binaries=True,
    name='多开管理器',
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
    exe_main,
    exe_multi,
    a_main.binaries,
    a_main.zipfiles,
    a_main.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cannotmax',
)
