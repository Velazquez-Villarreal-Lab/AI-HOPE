@echo off
setlocal

echo ----------------------------------------
echo   Python 3.12.3 + Ollama Setup for Windows
echo ----------------------------------------

:: 1. Download and install Python 3.12.3
echo 🐍 Downloading Python 3.12.3 installer...
curl -LO https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe

echo 🔧 Installing Python 3.12.3...
start /wait python-3.12.3-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

del python-3.12.3-amd64.exe
echo ✅ Python 3.12.3 installation completed.

:: Refresh environment variables
echo 🔄 Refreshing environment...
setx PATH "%PATH%" >nul
for /f "delims=" %%i in ('where py') do set PYTHON_PATH=%%i

:: Confirm Python version
echo Current Python version:
py --version

:: 2. Install pip packages from requirements.txt
IF EXIST requirements.txt (
    echo 📦 Installing Python packages from requirements.txt...
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
) ELSE (
    echo ❌ requirements.txt not found in the current directory!
)


:: 3. Download and install Ollama for Windows
where ollama >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo 🤖 Downloading Ollama installer for Windows...
    curl -LO https://ollama.com/download/OllamaSetup.exe
    echo 🛠 Installing Ollama...
    start /wait OllamaSetup.exe
    del OllamaSetup.exe
) ELSE (
    echo ✅ Ollama is already installed.
)

echo ----------------------------------------
echo ✅ All done! Python 3.12.3 and Ollama are ready.
pause