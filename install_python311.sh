#!/bin/bash
# Install Python 3.11 untuk TensorFlow compatibility (Arch Linux)

echo "════════════════════════════════════════════════════════════════"
echo "  🐍 Python 3.11 Installation for TensorFlow"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "📋 Why Python 3.11?"
echo "   • Full TensorFlow support via pip"
echo "   • Best compatibility with ML libraries"
echo "   • Stable and well-tested"
echo ""

# Check current Python version
CURRENT_PY=$(python --version 2>&1 | awk '{print $2}')
echo "Current system Python: $CURRENT_PY"
echo ""

# Check if Python 3.11 already exists
if command -v python3.11 &>/dev/null; then
    PY311_VER=$(python3.11 --version 2>&1 | awk '{print $2}')
    echo "✅ Python 3.11 is already installed: $PY311_VER"
    echo ""
    echo "You can now run setup_venv.sh to create venv with Python 3.11"
    exit 0
fi

echo "📦 Installing Python 3.11..."
echo ""

# Check available in repos
if pacman -Si python311 &>/dev/null; then
    echo "Installing from official repos..."
    sudo pacman -S python311 --noconfirm
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Python 3.11 installed successfully!"
        python3.11 --version
    else
        echo ""
        echo "❌ Installation failed!"
        exit 1
    fi
else
    echo "⚠️  python311 package not found in repos"
    echo ""
    echo "Alternative options:"
    echo ""
    echo "1. Use AUR:"
    echo "   yay -S python311"
    echo ""
    echo "2. Use pyenv:"
    echo "   curl https://pyenv.run | bash"
    echo "   pyenv install 3.11.9"
    echo ""
    echo "3. Use conda:"
    echo "   conda create -n automata python=3.11"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🎉 Installation complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Create virtual environment with Python 3.11:"
echo "   ./setup_venv.sh"
echo ""
echo "2. The script will automatically detect and use Python 3.11"
echo ""
