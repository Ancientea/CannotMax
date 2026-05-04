@echo off
if exist "%~dp0cannotmax.exe" (
    "%~dp0cannotmax.exe" multi
) else (
    uv run cannotmax multi
)