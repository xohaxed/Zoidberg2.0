@echo off
REM Launch Pneumonia Detection AI Dashboard
REM This script activates the virtual environment and starts the Streamlit dashboard

echo ========================================
echo  Pneumonia Detection AI Dashboard
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [2/3] Installing Streamlit...
    pip install streamlit plotly -q
) else (
    echo [2/3] Streamlit already installed
)

REM Launch dashboard
echo [3/3] Launching dashboard...
echo.
echo Dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run dashboard.py

pause
