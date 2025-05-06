#!/bin/bash

# Exit immediately on error
set -e

# Detect shell config file
SHELL_CONFIG=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    echo "❌ Unsupported shell. Please use bash or zsh."
    exit 1
fi

echo "🔧 Starting setup..."

# Install Homebrew if needed
if ! command -v brew &>/dev/null; then
    echo "🛠 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed."
fi

# Install Tcl-Tk and pyenv
echo "📦 Installing tcl-tk and pyenv..."
brew install tcl-tk pyenv

# Add pyenv and Tcl-Tk env config to shell profile
if ! grep -q 'pyenv init' "$SHELL_CONFIG"; then
    {
        echo ''
        echo '# >>> pyenv and Tcl-Tk setup >>>'
        echo 'export PYENV_ROOT="$HOME/.pyenv"'
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
        echo 'export PATH="/opt/homebrew/opt/tcl-tk/bin:$PATH"'
        echo 'export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"'
        echo 'export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"'
        echo 'export PKG_CONFIG_PATH="/opt/homebrew/opt/tcl-tk/lib/pkgconfig"'
        echo 'eval "$(pyenv init --path)"'
        echo 'eval "$(pyenv init -)"'
        echo 'eval "$(pyenv virtualenv-init -)"'
        echo '# <<< pyenv and Tcl-Tk setup <<<'
    } >> "$SHELL_CONFIG"
    echo "✅ Environment config added to $SHELL_CONFIG"
fi

# Load environment for current session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="/opt/homebrew/opt/tcl-tk/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/tcl-tk/lib/pkgconfig"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.12.3 with Tkinter support
if ! pyenv versions | grep -q "3.12.3"; then
    echo "🐍 Installing Python 3.12.3 (with Tkinter support)..."
    env \
      CPPFLAGS="$CPPFLAGS" \
      LDFLAGS="$LDFLAGS" \
      PKG_CONFIG_PATH="$PKG_CONFIG_PATH" \
      pyenv install 3.12.3
else
    echo "✅ Python 3.12.3 already installed."
fi

# Set Python 3.12.3 as global
pyenv global 3.12.3
hash -r

# Add alias for python
if ! grep -q 'alias python=' "$SHELL_CONFIG"; then
    echo "alias python=\"$HOME/.pyenv/versions/3.12.3/bin/python3.12\"" >> "$SHELL_CONFIG"
    echo "✅ Alias for python added to $SHELL_CONFIG"
fi

# Apply config
source "$SHELL_CONFIG"

# Confirm python version
echo "🐍 Python version: $(python --version)"
python -c "import tkinter; print('✅ Tkinter is installed. Version:', tkinter.TkVersion)"

# Upgrade pip
pip install --upgrade pip

# Install Ollama and pull llama3
if ! command -v ollama &>/dev/null; then
    echo "🤖 Installing Ollama..."
    brew install ollama
    echo "✅ Ollama installed."
else
    echo "✅ Ollama already installed."
fi

if ! ollama list | grep -q '^llama3:'; then
    echo "📥 Pulling llama3 model..."
    ollama pull llama3
    echo "✅ llama3 model pulled."
else
    echo "✅ llama3 model already present."
fi

# Install Python requirements
if [[ -f requirements.txt ]]; then
    echo "📦 Installing Python packages from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Python packages installed."
else
    echo "⚠️ requirements.txt not found. Skipping package installation."
fi

echo "🎉 Setup complete! Python 3.12.3, Tkinter, and llama3 are ready to use."