# Kaggle Integration - Quick Start Guide

## 🚀 Setup Kaggle API (One-time)

### Method 1: Via UI (Recommended)
```bash
python ui.py
# Pilih option 4 atau 3
# UI akan guide setup credentials
```

### Method 2: Manual Setup
```bash
# 1. Download kaggle.json dari kaggle.com/account
# 2. Copy ke ~/.kaggle/
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 📥 Download Dataset Only

**Workflow:** Download → Use in DSL manually

```bash
python ui.py
```

```
Choose option: 4  # Download Dataset from Kaggle

📚 Popular datasets:
  1. uciml/iris - Iris Species Dataset
  2. mlg-ulb/creditcardfraud - Credit Card Fraud
  3. heptapod/titanic - Titanic Dataset
  4. uciml/pima-indians-diabetes-database - Diabetes
  5. uciml/breast-cancer-wisconsin-data - Breast Cancer
  6. Enter custom dataset path

Choose dataset: 1  # Iris

⏳ Downloading uciml/iris...
✅ Dataset downloaded successfully!

📁 Location: datasets/
📄 CSV files found:
  • Iris.csv (4.5 KB)
```

Dataset siap digunakan di `datasets/Iris.csv`

## 🤖 Auto-Generate (SMART!)

**Workflow:** Download → Analyze → Generate DSL → Compile → Run

```bash
python ui.py
```

```
Choose option: 3  # Auto-Generate DSL from Kaggle

📚 Popular datasets:
  1. uciml/iris
  2. mlg-ulb/creditcardfraud
  3. heptapod/titanic
  4. uciml/pima-indians-diabetes-database
  5. uciml/breast-cancer-wisconsin-data
  6. Enter custom dataset path

Choose dataset: 1

⏳ Step 1/4: Downloading dataset...
✅ Downloaded: Iris.csv (4.5 KB)

⏳ Step 2/4: Analyzing Iris.csv...
✅ Analysis complete!
   • Rows: 150
   • Columns: 5
   • Target: Species
   • Task: classification
   • Classes: 3

⏳ Step 3/4: Generating DSL code...
✅ DSL generated!

⏳ Step 4/4: Saving...
✅ DSL file saved: auto_generated_irisclassifier.sndsl

💡 Next steps:
   1. View generated DSL (option 5)
   2. Validate (option 8)
   3. Compile & Run (option A)
```

Lanjut dengan:
```
Choose option: 5  # View DSL

Choose option: 8  # Validate

Choose option: A  # Compile & Run
```

## 📊 Example Generated DSL

File: `examples/auto_generated_irisclassifier.sndsl`

```dsl
# Auto-generated from Iris.csv
# Task: classification (3 classes)
# Dataset: 150 rows × 5 columns

DATASET load "datasets/Iris.csv" TARGET "Species"

MODEL "IrisClassifier" {
    # Input layer
    LAYER DENSE units: 64 activation: "relu"
    LAYER DROPOUT rate: 0.2
    
    # Hidden layer
    LAYER DENSE units: 32 activation: "relu"
    LAYER DROPOUT rate: 0.2
    
    # Output layer (3 classes)
    LAYER DENSE units: 3 activation: "softmax"
    
    OPTIMIZER "adam" lr: 0.001
    TRAIN epochs: 100 batch_size: 16
}
```

## 🎯 Full Workflow Example

### Scenario: Titanic Survival Prediction

```bash
python ui.py
```

**Step 1: Auto-generate**
```
Choose option: 3
Choose dataset: 3  # Titanic
Model name: TitanicPredictor
Target column: Survived
```

**Step 2: View DSL**
```
Choose option: 5
```

**Step 3: Edit (if needed)**
```
Choose option: 2  # Write DSL
# Edit epochs, layers, etc.
```

**Step 4: Validate**
```
Choose option: 8
```

**Step 5: Compile & Run**
```
Choose option: A
```

**Output:**
```
⏳ Compiling...
✅ Python code generated: output/titanicpredictor_model.py

⏳ Running...
Epoch 1/100: loss: 0.6234 - accuracy: 0.7654
Epoch 2/100: loss: 0.5123 - accuracy: 0.7891
...
Epoch 100/100: loss: 0.3456 - accuracy: 0.8765

✅ Training complete!

📊 RESULTS:
   • Accuracy: 87.65%
   • Loss: 0.3456
   • Model saved: models/titanicpredictor.h5
```

## 🛠️ Custom Dataset

### From Kaggle Search

1. Buka [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Search dataset (e.g., "wine quality")
3. Copy path dari URL: `uciml/wine-quality-data-set`

```bash
python ui.py
```

```
Choose option: 4  # Download
Choose dataset: 6  # Custom
Enter dataset path: uciml/wine-quality-data-set
```

### Using Downloaded Dataset

```bash
python ui.py
```

```
Choose option: 2  # Write DSL

# Enter DSL:
DATASET load "datasets/winequality-red.csv" TARGET "quality"

MODEL "WineQualityPredictor" {
    LAYER DENSE units: 128 activation: "relu"
    LAYER DROPOUT rate: 0.3
    LAYER DENSE units: 64 activation: "relu"
    LAYER DENSE units: 1 activation: "linear"
    
    OPTIMIZER "adam" lr: 0.0001
    TRAIN epochs: 200 batch_size: 32
}
```

## 🔧 Troubleshooting

### Kaggle CLI Not Found
```bash
pip install kaggle
# or
sudo pacman -S python-kaggle
```

### Credentials Error
```bash
# Check if file exists
ls -la ~/.kaggle/kaggle.json

# Fix permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Dataset Not Found
- Verify dataset path: `owner/dataset-name`
- Check if dataset is public
- Try via web: kaggle.com/datasets/owner/dataset-name

### Download Timeout
- Check internet connection
- Try smaller dataset
- Manual download from website

## 📈 Popular Datasets by Category

### Classification
- `uciml/iris` - Beginner friendly, 3 classes
- `uciml/breast-cancer-wisconsin-data` - Binary classification
- `uciml/wine-quality-data-set` - Wine quality grading

### Regression
- `uciml/california-housing-prices` - House prices
- `fedesoriano/stroke-prediction-dataset` - Healthcare

### Time Series
- `robikscube/hourly-energy-consumption` - Energy forecasting

### Computer Vision (CSV metadata)
- `paultimothymooney/chest-xray-pneumonia` - Medical imaging

## 💡 Pro Tips

1. **Start with Iris**: Simplest dataset untuk testing
2. **Use Auto-Generate**: Faster than manual DSL writing
3. **Check Dataset Size**: Download besar butuh waktu
4. **Validate First**: Always validate before compile
5. **Save Credentials**: Setup sekali, pakai selamanya

## 🎓 Learning Path

1. **Beginner**: Iris (option 3) → View → Run
2. **Intermediate**: Titanic → Edit DSL → Custom layers
3. **Advanced**: Custom dataset → Complex architecture

## 📚 Resources

- Full guide: `docs/08-kaggle-integration.md`
- DSL syntax: `docs/04-grammar-token.md`
- Examples: `examples/*.sndsl`
- Main docs: `README.md`

## ⚡ One-Liner

```bash
# Download Iris, auto-generate DSL, compile & run
python ui.py
# 3 → 1 → [enter] → [enter] → A
```

## 🎉 Success Checklist

- [ ] Kaggle CLI installed (`kaggle --version`)
- [ ] API credentials setup (`~/.kaggle/kaggle.json`)
- [ ] UI running (`python ui.py`)
- [ ] Dataset downloaded (option 4)
- [ ] DSL auto-generated (option 3)
- [ ] Model trained successfully (option A)

Happy ML modeling! 🚀
