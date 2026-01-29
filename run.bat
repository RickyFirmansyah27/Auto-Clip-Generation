@echo off
title AI Auto Shorts
echo ========================================
echo        AI AUTO SHORTS GUI
echo ========================================
echo.

REM Set FFmpeg path
set FFMPEG_DIR=%~dp0ffmpeg-master-latest-win64-gpl
set PATH=%FFMPEG_DIR%\bin;%PATH%

REM Check if FFmpeg exists, if not download and extract
if not exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
    echo [INFO] FFmpeg not found. Downloading...
    echo.

    REM Download FFmpeg using curl (built into Windows 10+)
    curl -L -o "%~dp0ffmpeg.zip" "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

    if errorlevel 1 (
        echo [ERROR] Failed to download FFmpeg!
        echo Please download manually from: https://github.com/BtbN/FFmpeg-Builds/releases
        pause
        exit /b 1
    )

    echo [INFO] Extracting FFmpeg...

    REM Extract using PowerShell
    powershell -command "Expand-Archive -Path '%~dp0ffmpeg.zip' -DestinationPath '%~dp0' -Force"

    if errorlevel 1 (
        echo [ERROR] Failed to extract FFmpeg!
        pause
        exit /b 1
    )

    REM Clean up zip file
    del "%~dp0ffmpeg.zip"

    echo [SUCCESS] FFmpeg installed successfully!
    echo.
)

REM Check and install Python dependencies
if exist "%~dp0requirements.txt" (
    echo [INFO] Checking Python dependencies...
    python -m pip install -r "%~dp0requirements.txt" -q
    if errorlevel 1 (
        echo [WARNING] Some dependencies may have failed to install.
    ) else (
        echo [SUCCESS] Dependencies installed!
    )
    echo.
)

echo Starting application...
echo.
python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed or Python not found.
    echo Make sure Python is installed and added to PATH.
    pause
)
