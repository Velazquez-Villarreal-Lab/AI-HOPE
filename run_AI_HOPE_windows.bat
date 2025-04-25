@echo off
setlocal

echo ----------------------------------------
echo 🧠 Checking if llama3 model is available via Ollama...
echo ----------------------------------------

:: Check if llama3 model is already pulled
ollama list | findstr /i "llama3:" >nul

IF %ERRORLEVEL% EQU 0 (
    echo llama3 is already pulled!
) ELSE (
    echo llama3 model not found.
    echo This appears to be the first time running the app.
    echo Pulling llama3 model now using Ollama...
    ollama pull llama3
)

echo ----------------------------------------
echo Starting AI_HOPE.py...
py AI_HOPE.py

pause