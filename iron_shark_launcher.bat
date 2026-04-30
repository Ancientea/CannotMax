@echo off
chcp 65001 >nul
title CannotMax
echo 请按任意键启动CannotMax...
pause >nul

set "current_dir=%cd%"

:: 检查 uv 是否存在
where uv >nul 2>nul
if %errorlevel% equ 0 goto run_main

:: 安装 uv
echo 未检测到 uv，正在安装...
powershell -ExecutionPolicy Bypass -Command "irm https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.ps1 | iex"

call :refresh_path
:: 验证 uv 是否可用
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo 安装 uv 后仍未找到，请检查安装路径
    pause
    exit /b 1
)
:: ===================================

:run_main
cd /d "%current_dir%"

:: ===== 选择推理环境选项询问 =====
set "torch_choice=none"
echo.
echo 选择推理环境? (5秒后自动跳过)
echo   C/c - Pytorch CPU版本
echo   D/d - Pytorch CUDA 12.8版本
echo   E/e - Pytorch CUDA 13.0版本
echo   N/n - 使用onnxruntime（默认）
echo ------------------------------------

:: 使用choice命令实现带超时的输入
choice /c CDEN /t 5 /d N /n >nul
:: choice 的 errorlevel 对应顺序为: C=1, D=2, E=3, N=4
if errorlevel 4 (
    set "torch_choice=none"
) else if errorlevel 3 (
    set "torch_choice=cu130"
) else if errorlevel 2 (
    set "torch_choice=cu128"
) else (
    set "torch_choice=cpu"
)

:: 根据选择使用对应环境运行主程序
if "%torch_choice%"=="cpu" (
    echo 使用Pytorch CPU版本...
    uv sync --extra cpu
) else if "%torch_choice%"=="cu128" (
    echo 使用Pytorch CUDA 12.8版本...
    uv sync --extra cu128
) else if "%torch_choice%"=="cu130" (
    echo 使用Pytorch CUDA 13.0版本...
    uv sync --extra cu130
) else (
    echo 使用onnxruntime...
    uv sync
)
echo.

:: ===================================
uv run main.py

echo 主程序已退出，感谢您的使用！
pause >nul
exit /b

:: 刷新 PATH 的函数
:refresh_path
for /f "skip=2 tokens=3*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYSTEM_PATH=%%a %%b"
for /f "skip=2 tokens=3*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%a %%b"
set "PATH=%USER_PATH%;%SYSTEM_PATH%"
exit /b