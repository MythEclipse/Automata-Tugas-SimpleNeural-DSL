#!/bin/bash
# Install TensorFlow untuk Python 3.14 (Arch Linux)

echo "════════════════════════════════════════════════════════════════"
echo "  📦 TensorFlow Installation for Python 3.14"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"
echo ""

if [[ "$PYTHON_VERSION" == "3.14" ]]; then
    echo "⚠️  Python 3.14 detected"
    echo "TensorFlow is not available via pip for Python 3.14+"
    echo ""
    echo "Installing via pacman (system package)..."
    echo ""
    
    # Check if already installed
    if pacman -Qi python-tensorflow &>/dev/null; then
        echo "✅ python-tensorflow is already installed!"
        TF_VERSION=$(python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null)
        echo "   Version: $TF_VERSION"
    else
        echo "📦 Installing python-tensorflow..."
        sudo pacman -S python-tensorflow --noconfirm
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ TensorFlow installed successfully!"
            TF_VERSION=$(python -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null)
            echo "   Version: $TF_VERSION"
        else
            echo ""
            echo "❌ Installation failed!"
            echo "Please try manually: sudo pacman -S python-tensorflow"
            exit 1
        fi
    fi
else
    echo "Installing via pip..."
    pip install tensorflow
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ VERIFICATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Verify installation
if python -c "import tensorflow" 2>/dev/null; then
    echo "✅ TensorFlow import successful!"
    python -c "import tensorflow as tf; print(f'   TensorFlow version: {tf.__version__}')"
    python -c "import tensorflow as tf; print(f'   Keras version: {tf.keras.__version__}')"
else
    echo "❌ TensorFlow import failed!"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🎉 Setup complete! You can now use TensorFlow."
echo "════════════════════════════════════════════════════════════════"
echo ""
