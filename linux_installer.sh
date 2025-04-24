#!/bin/bash

# Exit on any error
set -e

# Detect shell config file
SHELL_CONFIG=""
if [[ "$SHELL" == */bash ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [[ "$SHELL" == */zsh ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
else
    echo "❌ Unsupported shell. Please use bash or zsh."
    exit 1
fi

echo "🔧 Installing prerequisites..."

# Install required build dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    wget \
    curl \
    git

# Install pyenv if not installed
if ! command -v pyenv &>/dev/null; then
    echo "📦 Installing pyenv..."
    curl https://pyenv.run | bash

    # Add pyenv init to shell config
    if ! grep -q 'pyenv init' "$SHELL_CONFIG"; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_CONFIG"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv init --path)"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv init -)"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv virtualenv-init -)"' >> "$SHELL_CONFIG"
        echo "✅ pyenv config added to $SHELL_CONFIG"
    fi

    # Load pyenv immediately for this session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
else
    echo "✅ pyenv already installed."
fi

# Install Python 3.12.3
if ! pyenv versions | grep -q "3.12.3"; then
    echo "🐍 Installing Python 3.12.3 with pyenv..."
    pyenv install 3.12.3
fi

# Set Python 3.12.3 as global
pyenv global 3.12.3
hash -r  # Refresh shell cache

# Add alias so python points to newly installed python3.12
if ! grep -q 'alias python=' "$SHELL_CONFIG"; then
    echo "alias python=\"$HOME/.pyenv/versions/3.12.3/bin/python3.12\"" >> "$SHELL_CONFIG"
    echo "✅ Alias added to $SHELL_CONFIG: python → python3.12"
fi

# Source updated shell config immediately
source "$SHELL_CONFIG"

# Confirm python version
echo "✅ Python now points to: $(which python)"
python --version

# Upgrade pip
pip install --upgrade pip

# Install Ollama if not installed
if ! command -v ollama &>/dev/null; then
    echo "🤖 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed."
else
    echo "✅ Ollama already installed."
fi

# Check for requirements.txt
if [ ! -f requirements.txt ]; then
    echo "❌ requirements.txt not found in current directory!"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python packages from requirements.txt..."
pip install -r requirements.txt
echo "✅ All packages installed."

echo "🎉 All set! Python 3.12.3 is now your default Python."
