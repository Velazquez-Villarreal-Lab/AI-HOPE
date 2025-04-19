#!/bin/bash

# Exit on error
set -e

# Detect shell
SHELL_CONFIG=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    echo "❌ Unsupported shell. Please use bash or zsh."
    exit 1
fi

echo "🔧 Installing prerequisites..."

# Install Homebrew if not installed
if ! command -v brew &>/dev/null; then
    echo "🛠 Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install pyenv if not installed
if ! command -v pyenv &>/dev/null; then
    echo "📦 Installing pyenv..."
    brew install pyenv
fi

# Ensure pyenv environment variables are in shell config
if ! grep -q 'pyenv init' "$SHELL_CONFIG"; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_CONFIG"
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_CONFIG"
    echo 'eval "$(pyenv init --path)"' >> "$SHELL_CONFIG"
    echo 'eval "$(pyenv init -)"' >> "$SHELL_CONFIG"
    echo 'eval "$(pyenv virtualenv-init -)"' >> "$SHELL_CONFIG"
    echo "✅ pyenv config added to $SHELL_CONFIG"
fi

# Load pyenv for current session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.12.3 if not already installed
if ! pyenv versions | grep -q "3.12.3"; then
    echo "🐍 Installing Python 3.12.3..."
    pyenv install 3.12.3
fi

# Set Python 3.12.3 globally
pyenv global 3.12.3
hash -r  # refresh shell command cache

# Confirm Python version
echo "✅ Python now points to: $(which python)"
python --version

# Upgrade pip
pip install --upgrade pip

# Install Ollama if not installed
if ! command -v ollama &>/dev/null; then
    echo "🤖 Installing Ollama..."
    brew install ollama
    echo "✅ Ollama installed."
else
    echo "✅ Ollama already installed."
fi

# Check for requirements.txt
if [ ! -f requirements.txt ]; then
    echo "❌ requirements.txt not found in current directory!"
    exit 1
fi

# Install Python packages
echo "📦 Installing Python packages from requirements.txt..."
pip install -r requirements.txt
echo "✅ All packages installed."

# Source shell config to make pyenv changes take effect immediately
echo "🔁 Sourcing your shell config ($SHELL_CONFIG)..."
source "$SHELL_CONFIG"

echo "🎉 All set! Python 3.12.3 is now your default Python."