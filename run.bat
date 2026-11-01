@echo off
REM Parakeet AI Desktop Prototype - Windows Launcher
REM Candidate: Rupali Goyal

echo ========================================================
echo   Launching Parakeet AI Desktop Prototype
echo   Candidate: Rupali Goyal
echo ========================================================
python main.py
if errorlevel 1 (
    echo.
    echo An error occurred while launching. Press any key to exit.
    pause
)

