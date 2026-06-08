@echo off
:: Nimbus Diffusion — build + install
:: Run as Administrator (required to write to Program Files)

setlocal
cd /d "%~dp0"

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found. Install CMake and add to PATH.
    pause & exit /b 1
)
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found. Install Git and add to PATH.
    pause & exit /b 1
)

set BUILD_DIR=%~dp0build
set BUNDLE_SRC=%BUILD_DIR%\NimbusDiffusor.ofx.bundle
set BUNDLE_DST=C:\Program Files\Common Files\OFX\Plugins\NimbusDiffusor.ofx.bundle

echo [1/3] Configuring...
cmake -S . -B build -A x64 -G "Visual Studio 18 2026" 2>nul
if errorlevel 1 cmake -S . -B build -A x64
if errorlevel 1 (
    echo CMake configure failed.
    pause & exit /b 1
)

echo [2/3] Building Release...
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 (
    echo Build failed.
    pause & exit /b 1
)

echo [3/3] Installing...
mkdir "%BUNDLE_DST%\Contents\Win64"    2>nul
mkdir "%BUNDLE_DST%\Contents\Resources" 2>nul

copy /Y "%BUNDLE_SRC%\Contents\Win64\NimbusDiffusor.ofx" ^
        "%BUNDLE_DST%\Contents\Win64\NimbusDiffusor.ofx"

copy /Y "%~dp0NimbusDiffusion.png" ^
        "%BUNDLE_DST%\Contents\Resources\NimbusDiffusion.png"

if errorlevel 1 (
    echo.
    echo ERROR: Could not write to Program Files. Run this script as Administrator.
    pause & exit /b 1
)

echo.
echo Done! Restart DaVinci Resolve.
echo If sliders look wrong, clear the OFX cache first:
echo   Preferences ^> System ^> Memory and GPU ^> Clear OFX cache
pause
