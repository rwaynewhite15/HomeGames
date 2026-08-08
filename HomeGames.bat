@echo off
title HomeGames Server
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 homegames_server.py
) else (
    python homegames_server.py
)
pause
