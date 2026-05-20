@echo off
setlocal EnableDelayedExpansion
title Laparoscopy Defogging AI - Launcher

REM ============================================================
REM  Laparoscopy Image Defogging AI  -  Windows Auto-Launcher
REM ============================================================

echo.
echo ============================================================
echo  Laparoscopy Defogging AI  -  Auto-Setup and Launch
echo ============================================================
echo.

REM --- Locate the project root ---
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if exist "%SCRIPT_DIR%\Laparoscopy\app.py" (
    set "PROJECT_DIR=%SCRIPT_DIR%\Laparoscopy"
) else if exist "%SCRIPT_DIR%\app.py" (
    set "PROJECT_DIR=%SCRIPT_DIR%"
) else if exist "%SCRIPT_DIR%\Laparoscopy Zip\Laparoscopy\app.py" (
    set "PROJECT_DIR=%SCRIPT_DIR%\Laparoscopy Zip\Laparoscopy"
) else (
    echo [ERROR] Cannot find app.py. Make sure launch.bat is placed
    echo         inside or next to the project folder.
    pause
    exit /b 1
)

echo [INFO] Project folder: "%PROJECT_DIR%"
echo.

REM --- Check Python ---
echo [STEP 1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)
echo [OK] Python found.
echo.

REM --- Check for virtual environment ---
echo [STEP 2/5] Checking for manual virtual environment...

set "VENV_DIR="
if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    set "VENV_DIR=%PROJECT_DIR%\.venv"
    echo [OK] Existing venv found ^(.venv^).
) else if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
    set "VENV_DIR=%PROJECT_DIR%\venv"
    echo [OK] Existing venv found ^(venv^).
) else (
    echo [ERROR] Virtual environment not found at "%PROJECT_DIR%\.venv" or "%PROJECT_DIR%\venv".
    echo         Please set up the venv manually before running this script.
    pause
    exit /b 1
)
echo.

REM --- Activate venv ---
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Virtual environment activated.
echo.



REM --- Check for model checkpoint ---
echo [STEP 3/3] Checking model checkpoint...
set "MODEL_PATH=%PROJECT_DIR%\scripts\checkpoints\pix2pix_laparoscopy_dc\best_net_G.pth"

if exist "%MODEL_PATH%" (
    echo [OK] Model checkpoint found!
    goto MODEL_CHECK_DONE
)

echo.
echo WARNING: Model file NOT found!
echo Expected: "%MODEL_PATH%"
echo The web interface will still launch but defogging will NOT work.
echo.
set /p CONTINUE_OPT="Continue launching anyway? (Y/N) [default=Y]: "
if /i "!CONTINUE_OPT!"=="N" (
    echo Exiting.
    pause
    exit /b 0
)

:MODEL_CHECK_DONE
echo.

REM --- Launch Flask server ---
:SERVER_LOOP
cls
echo ============================================================
echo  Laparoscopy Defogging AI - Server is Running
echo ============================================================
echo.
echo  Keep this window open to keep the website active.
echo  Press CTRL+C in this window to stop the server manually.
echo.
echo  Open your browser at: http://127.0.0.1:5000
echo ============================================================
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"

cd /d "%PROJECT_DIR%"
python app.py

echo.
echo ============================================================
echo  Server stopped or crashed.
echo ============================================================
echo.
echo  [R] Restart Server
echo  [E] Exit and Close
echo.
set /p POST_ACTION="Enter choice (R/E) [default=E]: "
if /i "!POST_ACTION!"=="R" goto SERVER_LOOP

echo Exiting...
exit /b 0
