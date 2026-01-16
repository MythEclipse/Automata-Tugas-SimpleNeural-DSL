# Setup Kaggle untuk Arch Linux

## Masalah

```bash
pip install kaggle
# error: externally-managed-environment
```

Arch Linux menggunakan system yang tidak mengizinkan install package Python langsung via pip untuk mencegah konflik dengan package manager.

## ✅ Solusi (Pilih Salah Satu)

### Option 1: Virtual Environment (RECOMMENDED) 🌟

**Quick Setup:**
```bash
# Jalankan setup script otomatis
./setup_venv.sh
```

**Manual Setup:**
```bash
# 1. Buat virtual environment
python -m venv ~/venv-automata

# 2. Activate
source ~/venv-automata/bin/activate

# 3. Install dependencies
pip install kaggle tensorflow pandas numpy scikit-learn

# 4. Jalankan UI
python ui.py

# 5. Setelah selesai, deactivate
deactivate
```

**Keuntungan:**
- ✅ Aman, tidak merusak system Python
- ✅ Isolated, tidak conflict dengan package lain
- ✅ Best practice untuk development
- ✅ Mudah di-delete jika ada masalah

**Tips:**
Tambahkan alias ke `~/.bashrc` atau `~/.zshrc`:
```bash
alias automata='source ~/venv-automata/bin/activate && cd ~/automata'
```

Lalu cukup ketik: `automata` untuk langsung activate dan masuk folder!

---

### Option 2: Pipx (For CLI Tools)

```bash
# 1. Install pipx
sudo pacman -S python-pipx

# 2. Install kaggle
pipx install kaggle

# 3. Install TensorFlow dll tetap pakai venv (lihat Option 1)
```

**Keuntungan:**
- ✅ Untuk CLI tools seperti kaggle
- ✅ Auto-managed virtual environment

**Kekurangan:**
- ⚠️ Hanya untuk CLI, tetap butuh venv untuk TensorFlow

---

### Option 3: System-wide (NOT RECOMMENDED) ⚠️

```bash
pip install kaggle --break-system-packages
```

**Keuntungan:**
- ✅ Cepat, langsung bisa pakai

**Kekurangan:**
- ❌ Risk merusak system Python
- ❌ Bisa conflict dengan pacman packages
- ❌ Tidak recommended oleh Arch Linux

**Gunakan hanya jika:**
- Testing cepat
- Temporary setup
- Paham risikonya

---

## 🚀 Quick Start (Virtual Environment)

### Setup Sekali (One-time)

```bash
# Method 1: Auto-setup
./setup_venv.sh

# Method 2: Manual
python -m venv ~/venv-automata
source ~/venv-automata/bin/activate
pip install -r requirements.txt  # Jika ada
# atau
pip install kaggle tensorflow pandas numpy scikit-learn
```

### Setiap Kali Pakai

```bash
# 1. Activate venv
source ~/venv-automata/bin/activate

# 2. Jalankan UI
python ui.py

# 3. Setelah selesai
deactivate
```

---

## 📋 Verifikasi Instalasi

```bash
# Activate venv dulu
source ~/venv-automata/bin/activate

# Check installations
kaggle --version
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import pandas; print(pandas.__version__)"

# Jika semua OK, jalankan UI
python ui.py
```

Di menu UI, option 3 dan 4 sekarang akan menunjukkan status:
- ✅ = Kaggle tersedia
- ⚠️ = Kaggle tidak tersedia (klik untuk instruksi install)

---

## 🔧 Troubleshooting

### "kaggle: command not found" setelah install

**Solusi:**
```bash
# Pastikan venv active
source ~/venv-automata/bin/activate

# Check PATH
which kaggle
# Seharusnya: /home/asephs/venv-automata/bin/kaggle
```

### "ModuleNotFoundError: No module named 'kaggle'"

**Solusi:**
```bash
# Install di venv yang active
source ~/venv-automata/bin/activate
pip install kaggle
```

### Lupa activate venv

**Gejala:** Import error, command not found

**Solusi:**
```bash
source ~/venv-automata/bin/activate
```

**Tips:** Buat alias seperti di atas!

---

## 🎯 Workflow Lengkap

```bash
# ============================================
# SETUP PERTAMA KALI (One-time)
# ============================================

# 1. Masuk folder project
cd ~/automata

# 2. Setup venv (pilih salah satu):
./setup_venv.sh              # Auto
# atau
python -m venv ~/venv-automata  # Manual
source ~/venv-automata/bin/activate
pip install kaggle tensorflow pandas numpy scikit-learn

# 3. Setup Kaggle API credentials
# Buka: https://www.kaggle.com/account
# Download kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# ============================================
# SETIAP KALI PAKAI
# ============================================

# 1. Activate venv
source ~/venv-automata/bin/activate

# 2. Jalankan UI
python ui.py

# 3. Pilih option 3 (Auto-generate)
# 4. Select dataset dan go!

# 5. Setelah selesai
deactivate
```

---

## 💡 Best Practices

1. **Selalu gunakan virtual environment** untuk Python projects
2. **Jangan pakai --break-system-packages** kecuali terpaksa
3. **Activate venv sebelum run** script apapun
4. **Deactivate setelah selesai** untuk kembali ke system Python
5. **Gunakan alias** untuk kemudahan (lihat tips di atas)

---

## 📚 Resources

- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [Arch Wiki - Python](https://wiki.archlinux.org/title/Python)
- [PEP 668 - Externally Managed Environments](https://peps.python.org/pep-0668/)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

---

## ✅ Summary

**Untuk Arch Linux:**
1. Gunakan virtual environment (`./setup_venv.sh`)
2. Activate sebelum pakai (`source ~/venv-automata/bin/activate`)
3. Jalankan UI (`python ui.py`)
4. Deactivate setelah selesai (`deactivate`)

**UI sekarang support:**
- ✅ Auto-detect Kaggle availability
- ⚠️ Show warning jika Kaggle tidak tersedia
- 📋 Show installation instructions di UI langsung
- 🚀 Graceful degradation - fitur lain tetap jalan

Happy coding! 🎉
