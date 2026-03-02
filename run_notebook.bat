@echo off
REM ============================================================================
REM ZOIDBERG2.0 - Run Specific Notebook
REM Usage: run_notebook.bat [notebook_number]
REM Example: run_notebook.bat 1  (runs notebook 01)
REM ============================================================================

setlocal

if "%1"=="" (
    echo Usage: run_notebook.bat [notebook_number]
    echo.
    echo Available notebooks:
    echo   1 - Data Analysis and Integrity Check
    echo   2 - Preprocessing and Feature Engineering
    echo   3 - Baseline Modeling
    echo   4 - Deep Learning CNN
    echo   5 - Comprehensive Evaluation ROC-AUC
    echo   6 - Model Interpretation Explainability
    echo   7 - Ensemble Methods Optimization
    echo.
    echo Example: run_notebook.bat 1
    pause
    exit /b 1
)

REM Map notebook number to file
if "%1"=="1" set NOTEBOOK=01_Data_Analysis_Integrity_Check.ipynb
if "%1"=="2" set NOTEBOOK=02_Preprocessing_FeatureEngineering.ipynb
if "%1"=="3" set NOTEBOOK=03_Baseline_Modeling.ipynb
if "%1"=="4" set NOTEBOOK=04_Deep_Learning_CNN.ipynb
if "%1"=="5" set NOTEBOOK=05_Comprehensive_Evaluation_ROC_AUC.ipynb
if "%1"=="6" set NOTEBOOK=06_Model_Interpretation_Explainability.ipynb
if "%1"=="7" set NOTEBOOK=07_Ensemble_Methods_Optimization.ipynb

if "%NOTEBOOK%"=="" (
    echo [ERROR] Invalid notebook number: %1
    echo Please use 1-7
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\" (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Running: %NOTEBOOK%
echo ========================================
echo.

REM Activate and run
call .venv\Scripts\activate.bat
jupyter notebook notebooks\%NOTEBOOK%

deactivate
endlocal
