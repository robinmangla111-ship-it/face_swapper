@echo off
echo ============================================================
echo   Face Swapper - Starting Application
echo ============================================================
echo.

cd /d "%~dp0\.."

REM Activate venv
call .venv\Scripts\activate.bat

echo Starting Backend API server on http://localhost:8000 ...
echo Starting Frontend on http://localhost:3000 ...
echo.

REM Start backend in background
start "Face Swapper Backend" cmd /c "cd backend && python main.py"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
if exist "frontend\package.json" (
    start "Face Swapper Frontend" cmd /c "cd frontend && npm run dev"
    echo.
    echo Application started!
    echo   Backend API: http://localhost:8000
    echo   Frontend UI: http://localhost:3000
    echo   API Docs:    http://localhost:8000/docs
) else (
    echo.
    echo Backend started!
    echo   Backend API: http://localhost:8000
    echo   API Docs:    http://localhost:8000/docs
    echo   Frontend not installed yet.
)

echo.
echo Press any key to stop...
pause >nul
