@echo off
setlocal enabledelayedexpansion
title Precision Crop Pathology & Diagnostic System
echo =====================================================================
echo   Precision Crop Pathology & Agronomic Decision Support System
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Verify Python is installed on the system
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not detected in your system PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Check and initialize virtual environment if missing
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not detected.
    echo [INFO] Step 1/2: Creating virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    
    echo [INFO] Step 2/2: Installing dependencies from requirements.txt...
    echo (This may take a few minutes on first run...)
    call .\.venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] Some dependencies had warnings during installation.
    )
    echo [INFO] Environment setup completed successfully!
    echo.
) else (
    call .\.venv\Scripts\activate.bat
)

:: 3. Check for model checkpoints and give a friendly reminder
if not exist "models\vit_best.pt" (
    echo ---------------------------------------------------------------------
    echo [NOTICE] Trained weights not found in the 'models\' directory:
    echo          - models\vit_best.pt
    echo          - models\localization_best.pt
    echo          - models\severity_best.pt
    echo.
    echo The web dashboard will open, but you must place your model files
    echo in 'models\' to perform foliar diagnosis and leaf localization.
    echo ---------------------------------------------------------------------
    echo.
)

:: 4. Launch the Streamlit application
echo [INFO] Starting Streamlit Web Application...
echo.
python -m streamlit run app.py

pause
