@echo off
cd /d "%~dp0"
echo ==============================================
echo Checking and Installing Python Dependencies...
echo ==============================================
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies. Please ensure Python and pip are installed correctly.
    pause
    exit /b
)
echo.
echo ==============================
echo Launching Streamlit Web App...
echo ==============================
python -m streamlit run app.py
pause
