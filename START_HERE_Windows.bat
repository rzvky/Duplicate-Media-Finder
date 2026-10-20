@echo off
setlocal EnableExtensions
title Duplicate Media Finder
cd /d "%~dp0"
if not exist "%~dp0dupefinder\__init__.py" goto notextracted
set "VENV=%~dp0.venv"
set "PYV=%VENV%\Scripts\python.exe"

if exist "%PYV%" goto ready

echo.
echo  Duplicate Media Finder: first-time setup
echo  This needs internet once and takes about 2 to 5 minutes.
echo.

rem Find the full path of an installed Python (3.12 is the best supported)
set "PYEXE="
for %%V in (3.12 3.13 3.11 3.10 3) do (
  if not defined PYEXE (
    for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable if sys.version_info >= (3, 9) else '')" 2^>nul`) do set "PYEXE=%%P"
  )
)
if not defined PYEXE (
  for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable if sys.version_info >= (3, 9) else '')" 2^>nul`) do set "PYEXE=%%P"
)
if not defined PYEXE goto nopython
if not exist "%PYEXE%" goto nopython

echo  Using Python: %PYEXE%
"%PYEXE%" -m venv "%VENV%"
if errorlevel 1 goto failed
"%PYV%" -m pip install --disable-pip-version-check -q --upgrade pip
"%PYV%" -m pip install --disable-pip-version-check "Pillow>=10" "numpy>=1.22" "pillow-heif>=0.16"
if errorlevel 1 goto failed
echo.
echo  Installing camera RAW support...
"%PYV%" -m pip install --disable-pip-version-check "rawpy>=0.19"
if errorlevel 1 echo  Note: camera RAW support could not be installed. Everything else still works.
echo.
echo  Installing the AI engine...
"%PYV%" -m pip install --disable-pip-version-check "onnxruntime>=1.16"
if errorlevel 1 echo  Note: AI matching could not be installed. Exact and resized copies are still found.
echo.
echo  Installing video support...
"%PYV%" -m pip install --disable-pip-version-check "av>=14"
if errorlevel 1 echo  Note: video support could not be installed. Photos are still checked.
echo.
"%PYV%" -m dupefinder check
echo.
echo  Setup finished.
echo.

:ready
rem Older installs: add video support once (skipped if it could not be installed before)
"%PYV%" -c "import av" >nul 2>nul
if not errorlevel 1 goto run
if exist "%VENV%\no_video_support.txt" goto run
echo  Adding video support, one time only...
"%PYV%" -m pip install --disable-pip-version-check -q "av>=14"
if errorlevel 1 echo Video support could not be installed. Delete the .venv folder to try again.> "%VENV%\no_video_support.txt"

:run
"%PYV%" -m dupefinder %*
echo.
pause
exit /b 0

:notextracted
echo.
echo  Please unzip first: right-click the downloaded zip file, choose "Extract All",
echo  then open the extracted folder and double-click START_HERE_Windows.bat there.
echo.
pause
exit /b 1

:nopython
echo  Python is not installed on this computer.
echo.
echo  1. A download page will open now. Download Python 3.12 for Windows.
echo  2. In the installer, tick "Add python.exe to PATH", then click Install Now.
echo  3. Double-click START_HERE_Windows.bat again.
echo.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:failed
echo.
echo  Setup did not finish. Check your internet connection, then double-click this file again.
rmdir /s /q "%VENV%" >nul 2>nul
pause
exit /b 1
