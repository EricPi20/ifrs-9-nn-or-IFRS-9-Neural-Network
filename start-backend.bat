@echo off
REM Quick start script for Neural Network Scorecard Backend (Windows)
REM This script activates the virtual environment and starts the backend server

cd /d "%~dp0nn-scorecard\backend"

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run the installation steps first:
    echo   1. cd nn-scorecard\backend
    echo   2. python -m venv venv
    echo   3. venv\Scripts\activate
    echo   4. pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate

REM Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ❌ Dependencies not installed!
    echo Please install dependencies:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo 🚀 Starting Neural Network Scorecard Backend...
echo 📍 Backend will be available at: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause

