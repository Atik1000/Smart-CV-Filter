@echo off
REM Smart CV Filter - Setup Script for Windows
REM This script sets up the development environment

echo Setting up Smart CV Filter...

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo.
    echo Creating .env file...
    copy .env.example .env
    echo Please edit .env and add your OpenAI API key
)

REM Create necessary directories
echo.
echo Creating directories...
if not exist chroma_db mkdir chroma_db
if not exist uploads mkdir uploads

echo.
echo Setup complete!
echo.
echo To start the application:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run: streamlit run apps/streamlit_app.py
echo.
echo To configure OpenAI API key:
echo   Edit .env file and add: OPENAI_API_KEY=your_key_here
echo.
pause
