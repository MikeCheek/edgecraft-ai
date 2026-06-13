@echo off
REM EdgeCraft AI Startup Script for Windows

echo.
echo 🚀 EdgeCraft AI - Local TinyML Studio
echo ======================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.10+
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ %PYTHON_VERSION% found
echo.

REM Check Node.js
echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 18+
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo ✅ Node.js %NODE_VERSION% found
echo.

REM Backend Setup
echo Setting up backend...
cd backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt -q
echo ✅ Backend dependencies installed
echo.

REM Frontend Setup
echo Setting up frontend...
cd ..\frontend

if not exist node_modules (
    echo Installing npm dependencies...
    call npm install -q
)
echo ✅ Frontend dependencies installed
echo.

echo ======================================
echo 🎉 Setup Complete!
echo ======================================
echo.
echo To start the application:
echo.
echo Terminal 1 - Backend:
echo   cd backend
echo   venv\Scripts\activate.bat
echo   python -m uvicorn app.main:app --reload --port 8000
echo.
echo Terminal 2 - Frontend:
echo   cd frontend
echo   npm run dev
echo.
echo Then open:
echo   Frontend: http://localhost:5173
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
