@echo off
echo Starting Arthashila Application...
python run_app.py
if errorlevel 1 (
    echo Error running Arthashila. Please check your Python installation.
    pause
) 