@echo off
setlocal

echo ----------------------------------------
echo 🧠 Checking if llama3 model is available via Ollama...
echo ----------------------------------------

:: Get list of ollama models and search for 'llama3'
ollama list | findstr /i "llama3:" >nul

IF %ERRORLEVEL% EQU 0 (
    echo ✅ llama3 is already pulled!
    echo 🚀 Running AI_HOPE.py...
    py AI_HOPE.py
) ELSE (
    echo ❌ llama3 model not found.
    echo 👉 Run the following command to pull it:
    echo     ollama run llama3
    echo Then re-run this script.
)

pause