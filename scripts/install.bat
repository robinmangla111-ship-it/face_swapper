@echo off
echo ============================================================
echo   Face Swapper - Installation Script
echo ============================================================
echo.

cd /d "%~dp0\.."

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.9+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/5] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install core dependencies
echo.
echo [2/5] Installing core dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    pause
    exit /b 1
)

REM Install PyTorch (CPU) + GFPGAN for face enhancement
echo.
echo [3/5] Installing PyTorch (CPU) + GFPGAN for face enhancement...
echo   (This may take a few minutes - ~500MB download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gfpgan
echo   Done (or skipped if already installed)

REM Install frontend dependencies
echo.
echo [4/5] Installing frontend dependencies...
if exist "frontend\package.json" (
    cd frontend
    npm install
    cd ..
) else (
    echo   frontend not ready yet, skipping...
)

REM Download AI models
echo.
echo [5/5] Downloading AI models...
python backend\models\download_models.py

echo.
echo ============================================================
echo   Installation Complete!
echo   Run 'scripts\start.bat' to start the application
echo ============================================================
pause
