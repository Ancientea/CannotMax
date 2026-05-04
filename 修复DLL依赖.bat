@echo off
chcp 65001 >nul
echo ========================================
echo   CannotMax DLL 依赖修复工具
echo ========================================
echo.

set "VENV_SCRIPTS=.venv\Scripts"
if not exist "%VENV_SCRIPTS%" (
    echo [错误] 未找到 .venv\Scripts 目录
    echo 请将此脚本放在项目根目录下运行
    pause
    exit /b 1
)

echo 正在复制 VC++ 运行时 DLL...
set "FIXED=0"

:: 尝试多个来源
for %%S in (
    "C:\Windows\System32"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python313"
) do (
    if exist "%%~S\vcruntime140.dll" (
        copy /y "%%~S\vcruntime140.dll" "%VENV_SCRIPTS%\" >nul 2>nul && set "FIXED=1"
        copy /y "%%~S\vcruntime140_1.dll" "%VENV_SCRIPTS%\" >nul 2>nul
        copy /y "%%~S\msvcp140.dll" "%VENV_SCRIPTS%\" >nul 2>nul
        goto :done
    )
)

:done
if "%FIXED%"=="1" (
    echo [完成] DLL 已复制到 %VENV_SCRIPTS%
    echo 现在可以双击启动脚本运行 CannotMax 了
) else (
    echo [失败] 未找到 vcruntime140.dll
    echo 请手动安装 Visual C++ Redistributable:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
)
pause
