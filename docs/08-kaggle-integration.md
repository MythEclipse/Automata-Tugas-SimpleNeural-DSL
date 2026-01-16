# Kaggle Dataset Integration

## Overview

SimpleNeural-DSL mendukung download dataset langsung dari Kaggle melalui Kaggle API. Fitur ini memudahkan pengguna untuk mendapatkan dataset populer tanpa perlu download manual.

## Prerequisites

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

atau via pacman (Arch Linux):

```bash
sudo pacman -S python-kaggle
```

### 2. Setup Kaggle API Credentials

Anda memerlukan API credentials dari Kaggle:

1. Login ke [Kaggle](https://www.kaggle.com)
2. Buka **Account Settings** (`https://www.kaggle.com/account`)
3. Scroll ke bagian **API** section
4. Klik **Create New API Token**
5. File `kaggle.json` akan terdownload

#### Manual Setup

Simpan `kaggle.json` di direktori yang tepat:

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```bash
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

#### Setup via UI

SimpleNeural-DSL UI juga menyediakan setup otomatis:
- Pilih opsi `3` (Download Dataset from Kaggle)
- Jika credentials belum ada, UI akan meminta input
- Masukkan username dan API key
- Credentials akan disimpan otomatis

## Usage

### Via Interactive UI

1. Jalankan UI:
```bash
python ui.py
```

2. Pilih opsi `3` - Download Dataset from Kaggle

3. Pilih dataset:
   - **Option 1-5**: Popular datasets (Iris, Credit Card Fraud, Titanic, dll)
   - **Option 6**: Custom dataset path

4. Dataset akan didownload ke folder `datasets/`

5. UI akan menampilkan:
   - Lokasi download
   - Daftar file CSV yang ditemukan
   - Ukuran masing-masing file

### Popular Datasets

UI menyediakan quick access ke dataset populer:

| No | Dataset | Path | Description |
|----|---------|------|-------------|
| 1 | Iris | `uciml/iris` | Iris species classification (150 samples, 3 classes) |
| 2 | Credit Card Fraud | `mlg-ulb/creditcardfraud` | Credit card fraud detection |
| 3 | Titanic | `heptapod/titanic` | Titanic survival prediction |
| 4 | Diabetes | `uciml/pima-indians-diabetes-database` | Diabetes diagnosis |
| 5 | Breast Cancer | `uciml/breast-cancer-wisconsin-data` | Breast cancer diagnosis |

### Custom Dataset

Untuk dataset custom:

1. Pilih opsi `6`
2. Masukkan dataset path dalam format: `owner/dataset-name`
3. Contoh: `uciml/wine-quality-data-set`

## Dataset Path Format

Kaggle dataset path menggunakan format:
```
owner/dataset-name
```

Cara mendapatkan path:
1. Buka halaman dataset di Kaggle
2. Lihat URL: `https://www.kaggle.com/datasets/owner/dataset-name`
3. Copy bagian `owner/dataset-name`

### Contoh:
- URL: `https://www.kaggle.com/datasets/uciml/iris`
- Path: `uciml/iris`

## File Organization

```
automata/
├── datasets/              # Dataset yang didownload
│   ├── Iris.csv
│   ├── train.csv
│   └── test.csv
├── examples/              # DSL example files
│   └── iris_classification.sndsl
└── output/               # Generated Python code
    └── iris_classification_model.py
```

## Example Workflow

### Complete ML Pipeline with Kaggle Dataset

```bash
# 1. Jalankan UI
python ui.py

# 2. Download dataset (option 3)
Choose option: 3
Choose dataset: 1  # Iris dataset

# 3. Write DSL code (option 2)
Choose option: 2

# DSL Code:
DATASET load "datasets/Iris.csv" TARGET "Species"

MODEL "IrisClassifier" {
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    LAYER DENSE units: 32 activation: "relu"
    LAYER DENSE units: 3 activation: "softmax"
    
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 100 batch_size: 16
}

# 4. Compile & Run (option 9)
Choose option: 9
```

## Error Handling

### Common Issues

**1. Kaggle CLI Not Found**
```
❌ Kaggle CLI not found!
💡 Install with: pip install kaggle
```

**Solution:**
```bash
pip install kaggle
```

**2. API Credentials Not Found**
```
⚠️  Kaggle API credentials not found!
```

**Solution:**
- Setup via UI (input username & key)
- Or manually place `kaggle.json` in `~/.kaggle/`

**3. Invalid Dataset Path**
```
❌ Download failed!
Error: 404 - Not Found
```

**Solution:**
- Verify dataset path is correct
- Check dataset exists and is public
- Format: `owner/dataset-name` (no spaces)

**4. Download Timeout**
```
❌ Download timeout (>5 minutes)
```

**Solution:**
- Check internet connection
- Try smaller dataset
- Manually download from Kaggle website

**5. Permission Error**
```
❌ Error: [Errno 13] Permission denied
```

**Solution:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## API Rate Limits

Kaggle API memiliki rate limits:
- **Hourly limit**: ~100 requests
- **Daily limit**: ~1000 requests

Jika melebihi limit:
```
❌ Error: 429 - Too Many Requests
```

Wait beberapa menit sebelum retry.

## Security Notes

### API Key Protection

**DO:**
- ✅ Keep `kaggle.json` permissions at 600
- ✅ Never commit `kaggle.json` to git
- ✅ Add `.kaggle/` to `.gitignore`

**DON'T:**
- ❌ Share your API key publicly
- ❌ Commit credentials to repository
- ❌ Use root/admin account keys

### .gitignore Entry

Pastikan `.gitignore` contains:
```gitignore
# Kaggle credentials
.kaggle/
kaggle.json

# Downloaded datasets
datasets/
*.csv
```

## Advanced Usage

### Download Multiple Datasets

```python
# Via Python script
from ui import SimpleNeuralUI

ui = SimpleNeuralUI()

datasets = [
    'uciml/iris',
    'uciml/wine-quality-data-set',
    'uciml/breast-cancer-wisconsin-data'
]

for dataset in datasets:
    # Call download function
    pass
```

### Direct Kaggle CLI Usage

```bash
# Download dataset
kaggle datasets download -d uciml/iris -p datasets/ --unzip

# List files in dataset
kaggle datasets files -d uciml/iris

# Search datasets
kaggle datasets list -s "wine quality"
```

## Benefits

1. **Convenience**: Download datasets tanpa browser
2. **Automation**: Integrate dalam workflow
3. **Consistency**: Sama dataset version untuk semua
4. **Speed**: CLI lebih cepat dari web download
5. **Scripting**: Easy untuk automation scripts

## Future Enhancements

Planned features:
- [ ] Dataset search functionality
- [ ] Dataset preview before download
- [ ] Multi-file dataset handling
- [ ] Automatic dataset-to-DSL template generation
- [ ] Cached downloads (skip if exists)
- [ ] Dataset version management
- [ ] Competition data download support

## Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Kaggle API GitHub](https://github.com/Kaggle/kaggle-api)

## Summary

Kaggle integration mempermudah workflow:

```
Download Dataset → Write DSL → Compile → Train Model
     (option 3)   →  (option 2) → (option 8) → (option 9)
```

**One-line workflow:**
```bash
python ui.py
# 3 → 1 → 2 → [write DSL] → 9
```

Happy dataset downloading! 🎉
