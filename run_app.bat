@echo off
title Crop Disease Detection System - Final Review
echo =====================================================================
echo  Starting Precision Crop Pathology & Agronomic Decision Support System
echo =====================================================================
echo.
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python -m streamlit run app.py
pause
