# Health Sensing Sleep Apnea Detection Pipeline

A complete machine learning pipeline for sleep apnea event detection and classification using nasal airflow, thoracic movement, and SpO2 signals.

## Project Structure

```
Scenario2_HealthSensing/
├── Data/                          # Raw sensor data (5 participants)
│   ├── AP01/
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
├── Dataset/
│   └── breathing_dataset.csv      # Processed training dataset
├── Visualizations/
│   └── AP*.pdf                    # Signal visualizations
├── scripts/
│   ├── vis.py                     # Visualization script
│   ├── create_dataset.py          # Dataset creation
│   ├── train_model.py             # Model training
│   └── io_utils.py                # Shared I/O utilities
├── metrics_lopo_*.json            # Training metrics
└── README.md                      # This file
```

## Data Format

Raw data files are semicolon-separated text files with European timestamps:

**Signal Files** (Flow, Thorac, SPO2):
```
[Header info]
Data:
timestamp; value
30.05.2024 20:59:00,031; 0.123
30.05.2024 20:59:00,062; 0.125
...
```

**Event Files** (Flow Events, Sleep profile):
```
[Metadata]
29.05.2024 21:33:57,246-21:34:33,496; 36; Hypopnea; N1
29.05.2024 22:57:42,574-22:57:54,485; 12; Obstructive Apnea; N1
...
```

## Scripts

### 1. Visualization (`scripts/vis.py`)

Generate PDF visualizations of signals with event overlays.

**Usage:**
```bash
python scripts/vis.py -name Data/AP01 --out Visualizations
python scripts/vis.py -name Data/AP02 --out Visualizations --downsample 5s
python scripts/vis.py -name Data/AP01 --max-hours 2
```

**Options:**
- `-name PATH`: Path to participant folder (required)
- `--out DIR`: Output directory (default: Visualizations)
- `--downsample FREQ`: Resample frequency (default: 1s)
- `--max-hours N`: Limit to first N hours
- `--no-filter`: Skip bandpass filtering

**Output:**
- `Visualizations/<participant>_visualization.pdf`

---

### 2. Dataset Creation (`scripts/create_dataset.py`)

Process raw signals and create labeled training dataset.

**Usage:**
```bash
# Full run
python scripts\scripts\create_dataset.py -in_dir Data -out_dir Dataset

# Dry-run (no file write)
python scripts\scripts\create_dataset.py -in_dir Data -out_dir Dataset --dry-run

# Single participant
python scripts\scripts\create_dataset.py -in_dir Data -out_dir Dataset --participant AP01

# Verbose output
python scripts\scripts\create_dataset.py -in_dir Data -out_dir Dataset --verbose
```

**Options:**
- `-in_dir DIR`: Input data directory (required)
- `-out_dir DIR`: Output directory (required)
- `--participant ID`: Process single participant (optional)
- `--dry-run`: Report without saving
- `--verbose`: Print processing details

**Processing Steps:**
1. Load signals (nasal airflow, thoracic movement, SpO2)
2. Apply bandpass filter (0.17–0.4 Hz)
3. Create 30-second windows with 50% overlap
4. Label windows using event overlap (>50% = positive event)
5. Resample to fixed length (967 samples)
6. Save to CSV with columns: `participant`, `start`, `end`, `label`, `nasal`, `thor`, `spo2`

**Output:**
- `Dataset/breathing_dataset.csv` (~5,300 samples)

---

### 3. Model Training (`scripts/train_model.py`)

Train CNN or Conv-LSTM models using Leave-One-Participant-Out (LOPO) cross-validation.

**Usage:**
```bash
# CNN model
python scripts\train_model.py -dataset Dataset\breathing_dataset.csv --model cnn --epochs 10 --batch 32

# Conv-LSTM model
python scripts\train_model.py -dataset Dataset\breathing_dataset.csv --model convlstm --epochs 10 --batch 64
```

**Options:**
- `-dataset FILE`: Path to breathing_dataset.csv (required)
- `--model {cnn,convlstm}`: Model architecture (default: cnn)
- `--epochs N`: Training epochs per fold (default: 10)
- `--batch N`: Batch size (default: 64)

**Architectures:**

**CNN:**
```
Conv1D(32) → ReLU → MaxPool(2)
→ Conv1D(64) → ReLU → MaxPool(2)
→ Flatten → Dense(128) → ReLU → Dense(n_classes)
```

**Conv-LSTM:**
```
Conv1D(32) → ReLU
→ Conv1D(64) → ReLU
→ LSTM(64) → Dense(128) → ReLU → Dense(n_classes)
```

**Output:**
- `metrics_lopo_cnn.json` or `metrics_lopo_convlstm.json`
- Contains accuracy, precision, recall, confusion matrices per fold

**Example Results (5 epochs):**
```
Fold 1 (AP01): 89.01% accuracy
Fold 2 (AP02): 91.64% accuracy
Fold 3 (AP03): 98.40% accuracy

Precision (Normal): 95.2% ± 2.9%
Recall (Normal):    97.6% ± 3.1%
```

---

### 4. I/O Utilities (`scripts/io_utils.py`)

Shared functions for robust signal and event file parsing. Used by both visualization and dataset creation scripts.

**Key Functions:**
- `read_signal_file(path)` - Parse semicolon-separated signal files with European timestamps
- `read_events(path)` - Parse event files with ranged or discrete events

**Features:**
- Automatic timestamp format detection and normalization
- Flexible separator handling (semicolon, comma, whitespace)
- European date format support (dd.mm.yyyy)
- Missing value handling

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- scipy
- matplotlib

### Setup

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install torch scikit-learn pandas numpy scipy matplotlib
```

---

## Workflow

### Step 1: Visualize Raw Data
```bash
python scripts\vis.py -name Data/AP01 --out Visualizations
```
Outputs: `Visualizations/AP01_visualization.pdf`

### Step 2: Create Dataset
```bash
python scripts\scripts\create_dataset.py -in_dir Data -out_dir Dataset --dry-run
python scripts\scripts\create_dataset.py -in_dir Data -out_dir Dataset
```
Outputs: `Dataset/breathing_dataset.csv`

### Step 3: Train Models
```bash
python scripts\train_model.py -dataset Dataset\breathing_dataset.csv --model cnn --epochs 10
python scripts\train_model.py -dataset Dataset\breathing_dataset.csv --model convlstm --epochs 10
```
Outputs: `metrics_lopo_cnn.json`, `metrics_lopo_convlstm.json`

---

## Key Classes

### WindowDataset
PyTorch Dataset class for loading windows with labels.
- Handles string-to-array conversion
- Supports per-fold label mapping for consistent class indices
- Returns (nasal, thoracic) 2-channel stacked tensors

### Simple1DCNN
1D Convolutional Neural Network for time-series classification.
- Input: (batch_size, 2_channels, 967_timesteps)
- Output: (batch_size, n_classes)

### ConvLSTM
Hybrid Conv-LSTM model combining CNNs and LSTMs.
- CNN extracts temporal features
- LSTM learns long-term dependencies
- Output: (batch_size, n_classes)

---

## Signal Details

| Signal     | Sampling Rate | Duration  | Channels | Use Case         |
|------------|---------------|-----------|----------|------------------|
| Nasal Flow | 32 Hz         | Full night| 1        | Primary detector |
| Thoracic   | 32 Hz         | Full night| 1        | Respiratory effort|
| SpO2       | 4 Hz          | Full night| 1        | Oxygen saturation|

**Window Configuration:**
- Duration: 30 seconds
- Overlap: 50%
- Resampled Length: 967 samples (match nasal at 32 Hz)
- Labels: Normal, Hypopnea, Obstructive Apnea

---

## Performance Notes

- **Best Fold Performance:** AP03 at 98.40% accuracy (balanced events)
- **Class Imbalance:** Dataset heavily skewed toward "Normal" class
- **Recommendation:** Use weighted loss functions or class balancing for production
- **Device:** CPU recommended for laptops; GPU recommended for large-scale training

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No valid data found` | Check event file format; ensure semicolon separators |
| `Memory error during training` | Reduce batch size (--batch 16) |
| `Low accuracy on minority classes` | Add class weights or use cost-sensitive learning |
| `File not found` | Verify participant folder exists in Data/ |

---

## Future Improvements

- [ ] Add data augmentation (time-shift, noise injection)
- [ ] Implement class balancing (weighted loss, SMOTE)
- [ ] Add hyperparameter tuning (grid search, Bayesian optimization)
- [ ] Export models to ONNX for production deployment
- [ ] Real-time inference API
- [ ] Multi-modal fusion (combine all 3 signals more effectively)

---

## License

Academic use only. Not for clinical deployment without validation.

---

## Contact & Support

For issues or questions, refer to the script docstrings and inline comments.# Scenario2_HealthSensing
