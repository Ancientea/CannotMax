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
powershell -ExecutionPolicy Bypass -Command "irm https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.ps1     | iex"

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

:: ===== PyTorch安装选项询问 =====
set "torch_choice=none"
echo.
echo 是否需要安装PyTorch? (5秒后自动跳过)
echo   C/c - CPU版本
echo   G/g - CUDA版本
echo   N/n - 跳过安装
echo ------------------------------------

:: 使用choice命令实现带超时的输入
choice /c CGN /t 5 /d N /n >nul
if errorlevel 3 (
    set "torch_choice=none"
) else if errorlevel 2 (
    set "torch_choice=cuda"
) else if errorlevel 1 (
    set "torch_choice=cpu"
)

:: 根据选择安装PyTorch
if "%torch_choice%"=="cpu" (
    echo 安装CPU版本的PyTorch...
    uv add torch torchvision --extra cpu
) else if "%torch_choice%"=="cuda" (
    echo 安装CUDA版本的PyTorch...
    uv add torch torchvision --extra cu128
) else (
    echo 跳过PyTorch安装
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