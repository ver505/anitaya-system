@echo off
title AniTaya Web Server - CSST 102
cd /d "%~dp0"
echo =====================================================================
echo   AniTaya - Agricultural Harvest and Income Prediction System
echo   Web-Based Predictive Machine Learning Project (CSST 102)
echo   Laguna State Polytechnic University - College of Computer Studies
echo =====================================================================
echo.
echo [1/2] Launching your web browser at http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
echo [2/2] Starting Python Flask Web Server...
echo.
echo The website is now active! Keep this window open while using the website.
echo.
python app.py
pause
