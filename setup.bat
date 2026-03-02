@echo off
REM ============================================================================
REM ZOIDBERG2.0 - Initial Setup Script
REM Creates virtual environment and installs all dependencies
REM ============================================================================

echo.
echo ========================================
echo   ZOIDBERG2.0 - Project Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Create virtual environment if it doesn't exist
if exist ".venv\" (
    echo.
    echo [2/4] Virtual environment already exists. Skipping creation...
) else (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)

REM Activate virtual environment
echo.
echo [3/4] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [3.5/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [4/4] Installing dependencies from requirements.txt...
echo This may take several minutes...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
)

REM Verify installation
echo.
echo ========================================
echo   Verifying Installation
echo ========================================
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, cv2, torch; print('\n[SUCCESS] All core packages installed successfully!\n'); print(f'NumPy: {numpy.__version__}'); print(f'Pandas: {pandas.__version__}'); print(f'Scikit-learn: {sklearn.__version__}'); print(f'PyTorch: {torch.__version__}')"

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To start the project:
echo   - Run: run.bat (opens Jupyter Notebook)
echo   - Or: run_notebook.bat [1-2] (opens specific notebook)
echo.
echo Project Structure:
echo   notebooks/   - Jupyter notebooks (Step 1, Step 2, ...)
echo   data/        - Dataset storage
echo   src/         - Source code modules
echo   models/      - Saved models
echo   reports/     - Analysis results and figures
echo.

deactivate
pause
