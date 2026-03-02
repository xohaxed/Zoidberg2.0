@echo off
REM ============================================================================
REM ZOIDBERG2.0 - Project Runner Script
REM Medical Imaging / Computer Aided Diagnosis - Pneumonia Detection
REM ============================================================================

echo.
echo ========================================
echo   ZOIDBERG2.0 - Pneumonia Detection
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then install dependencies: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check Python and packages
echo.
echo [2/3] Verifying installation...
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import numpy, pandas, sklearn, torch; print('Core packages: OK')" 2>nul
if errorlevel 1 (
    echo [WARNING] Some packages may be missing. Run: pip install -r requirements.txt
)

REM Run Jupyter Notebook
echo.
echo [3/3] Starting Jupyter Notebook...
echo.
echo Available notebooks:
echo   - 01_Data_Analysis_Integrity_Check.ipynb
echo   - 02_Preprocessing_FeatureEngineering.ipynb
echo   - 03_Baseline_Modeling.ipynb
echo   - 04_Deep_Learning_CNN.ipynb
echo   - 05_Comprehensive_Evaluation_ROC_AUC.ipynb
echo   - 06_Model_Interpretation_Explainability.ipynb
echo   - 07_Ensemble_Methods_Optimization.ipynb
echo.
echo Opening Jupyter Notebook interface...
echo Press Ctrl+C to stop the server when done.
echo.

jupyter notebook

REM Deactivate on exit
deactivate
