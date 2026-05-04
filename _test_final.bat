@echo off
chcp 65001 >nul
title CannotMax
echo 请按任意键启动 CannotMax...
pause >nul

set "current_dir=%cd%"

where uv >nul 2>nul
if %errorlevel% equ 0 goto run_main

echo 未检测到 uv，正在安装...
powershell -ExecutionPolicy Bypass -Command "irm https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.ps1 | iex"

call :refresh_path
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo 安装 uv 后仍未找到
    pause
    exit /b 1
)

:run_main
cd /d "%current_dir%"

set "torch_choice=none"
echo.
echo 选择 PyTorch 版本 (5秒后默认跳过):
echo   1 - CUDA 12.8 (NVIDIA)
echo   2 - CUDA 13.0 (NVIDIA)
echo   A - AMD DirectML (RX6000/7000)
echo   C - CPU 版本 (通用)
echo   N - 跳过安装
echo ------------------------------------

choice /c 12ACN /t 5 /d N /n >nul
if errorlevel 5 (
    set "torch_choice=none"
) else if errorlevel 4 (
    set "torch_choice=cpu"
) else if errorlevel 3 (
    set "torch_choice=directml"
) else if errorlevel 2 (
    set "torch_choice=cu130"
) else if errorlevel 1 (
    set "torch_choice=cu128"
)

if "%torch_choice%"=="cpu" (
    echo 安装 CPU 版本 PyTorch...
    call :sync_extra cpu
) else if "%torch_choice%"=="cu128" (
    echo 安装 CUDA 12.8 版本 PyTorch...
    call :sync_extra cu128
) else if "%torch_choice%"=="cu130" (
    echo 安装 CUDA 13.0 版本 PyTorch...
    call :sync_extra cu130
) else if "%torch_choice%"=="directml" (
    echo 安装 AMD DirectML 版本 PyTorch...
    call :sync_extra directml
    echo 安装 torch-directml...
    pip install torch-directml
) else (
    echo 跳过 PyTorch 安装
)
echo.

uv run main.py

echo 主程序已退出，感谢您的使用！
pause >nul
exit /b

:sync_extra
if exist uv.lock del uv.lock
uv sync --extra %1
if %errorlevel% equ 0 exit /b
if exist .venv rmdir /s /q .venv
if exist uv.lock del uv.lock
uv sync --extra %1
if %errorlevel% neq 0 (
    echo PyTorch 安装失败
    pause
)
exit /b

:refresh_path
for /f "skip=2 tokens=3*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYSTEM_PATH=%%a %%b"
for /f "skip=2 tokens=3*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%a %%b"
set "PATH=%USER_PATH%;%SYSTEM_PATH%"
exit /b
