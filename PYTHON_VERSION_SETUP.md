# Python Version Setup untuk SimpleNeural-DSL

## 🎯 Masalah

Python 3.14 tidak support TensorFlow via pip:
```bash
pip install tensorflow
# ERROR: Could not find a version that satisfies the requirement tensorflow
```

## ✅ Solusi: 3 Opsi

### Option 1: Install Python 3.11 (RECOMMENDED) 🌟

**Why Python 3.11?**
- ✅ Full TensorFlow support via pip
- ✅ Best ML library compatibility  
- ✅ Stable dan well-tested
- ✅ No system package dependency

**Quick Install:**
```bash
# Auto-install Python 3.11
./install_python311.sh

# Atau manual
sudo pacman -S python311  # Jika tersedia
# atau
yay -S python311  # Via AUR
```

**Lalu recreate venv:**
```bash
# Delete old venv
rm -rf ~/venv-automata

# Create new venv (auto-detect Python 3.11)
./setup_venv.sh
```

Script akan otomatis prioritize Python 3.11!

---

### Option 2: Use System TensorFlow (Current Setup)

**How it works:**
- Venv dibuat dengan `--system-site-packages`
- TensorFlow diinstall via pacman (system-wide)
- Venv bisa akses system packages

**Install TensorFlow:**
```bash
# Install system TensorFlow
sudo pacman -S python-tensorflow

# Verify
python -c "import tensorflow; print(tensorflow.__version__)"
```

**Keuntungan:**
- ✅ Works dengan Python 3.14
- ✅ No need additional Python version

**Kekurangan:**
- ⚠️ Depend on system package
- ⚠️ Versi TensorFlow ditentukan by pacman

---

### Option 3: Use Conda/Pyenv

**Pyenv:**
```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.11
pyenv install 3.11.9

# Set for project
cd ~/automata
pyenv local 3.11.9

# Create venv
python -m venv venv-automata
source venv-automata/bin/activate
pip install tensorflow kaggle pandas numpy scikit-learn
```

**Conda:**
```bash
# Create conda env
conda create -n automata python=3.11

# Activate
conda activate automata

# Install packages
conda install tensorflow kaggle pandas numpy scikit-learn
```

---

## 🚀 Quick Start Guide

### Scenario 1: Fresh Install (Recommended)

```bash
# 1. Install Python 3.11
./install_python311.sh

# 2. Setup venv (auto uses Python 3.11)
./setup_venv.sh

# 3. Activate
source ~/venv-automata/bin/activate

# 4. Verify
python --version  # Should be 3.11.x
python -c "import tensorflow; print(tensorflow.__version__)"

# 5. Run UI
python ui.py
```

### Scenario 2: Current Setup (Python 3.14)

```bash
# 1. Install system TensorFlow
sudo pacman -S python-tensorflow

# 2. Setup venv (with system packages)
./setup_venv.sh

# 3. Activate
source ~/venv-automata/bin/activate

# 4. Verify
python -c "import tensorflow; print(tensorflow.__version__)"

# 5. Run UI
python ui.py
```

---

## 🔍 Auto-Detection

Script `setup_venv.sh` otomatis detect Python versions:

**Priority Order:**
1. Python 3.11 ✅ (Best)
2. Python 3.12 ✅
3. Python 3.10 ✅
4. Python 3.9 ✅
5. Python 3.14 ⚠️ (Fallback, needs system TensorFlow)

**Detection Output:**
```bash
🔍 Detecting available Python versions...
   Found: python3.11 (version 3.11)
   ✅ Selected: python3.11 (TensorFlow compatible)
```

---

## 📊 Comparison

| Method | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **Python 3.11** | Full pip support, isolated, stable | Need separate Python | ⭐⭐⭐⭐⭐ Best |
| **System TensorFlow** | Simple, works with 3.14 | System dependency | ⭐⭐⭐ OK |
| **Conda** | Complete env management | Heavy, slow | ⭐⭐⭐ Good |
| **Pyenv** | Flexible, multiple versions | Setup complexity | ⭐⭐⭐⭐ Great |

---

## 🛠️ Troubleshooting

### "python311: command not found"

**Solution:**
```bash
# Check repos
pacman -Ss python311

# Install from AUR
yay -S python311

# Or use pyenv (see Option 3)
```

### "TensorFlow import failed"

**Check:**
```bash
# 1. Verify TensorFlow installed
pip list | grep tensorflow
# or
pacman -Q python-tensorflow

# 2. Check Python version
python --version

# 3. Test import
python -c "import tensorflow"
```

### Venv still uses Python 3.14

**Solution:**
```bash
# Delete venv
rm -rf ~/venv-automata

# Specify Python version explicitly
python3.11 -m venv ~/venv-automata

# Or let script auto-detect
./setup_venv.sh
```

---

## 📝 Summary

**Recommended Workflow:**

```bash
# One-time setup
./install_python311.sh     # Install Python 3.11
./setup_venv.sh            # Create venv (auto-detect 3.11)

# Every time use
source ~/venv-automata/bin/activate
python ui.py

# When done
deactivate
```

**Current Status:**
- ✅ `setup_venv.sh` - Auto-detects best Python version
- ✅ `install_python311.sh` - Install Python 3.11 helper
- ✅ `install_tensorflow.sh` - Install system TensorFlow helper

Pick the option that works best for you! 🎉
