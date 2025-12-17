@echo off
REM Quick start script for Neural Network Scorecard Frontend (Windows)
REM This script starts the frontend development server

cd /d "%~dp0nn-scorecard\frontend"

REM Check if node_modules exists
if not exist "node_modules" (
    echo ❌ Node modules not found!
    echo Please install dependencies first:
    echo   npm install
    pause
    exit /b 1
)

echo 🚀 Starting Neural Network Scorecard Frontend...
echo 📍 Frontend will be available at: http://localhost:5173
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the development server
npm run dev

pause

