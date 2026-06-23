# ZOIDBERG2.0

**Medical Imaging / Computer Aided Diagnosis - Pneumonia Detection**

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)](.)

---

## 📋 Project Overview

ZOIDBERG2.0 is an enterprise-grade Machine Learning pipeline for detecting Pneumonia from chest X-ray images. This project implements a rigorous, reproducible approach to medical image classification using multiple datasets and advanced ML techniques.

### Key Features

- ✅ **Multi-Dataset Architecture**: Uses 3 separate datasets with dedicated hyperparameter tuning set
- ✅ **Comprehensive Validation**: Compares Cross-Validation vs Train-Test Split strategies
- ✅ **Advanced Feature Engineering**: Implements PCA for dimensionality reduction
- ✅ **Multiple Algorithms**: Baseline models (SVM, Logistic Regression) and Deep Learning (CNNs)
- ✅ **Proper Evaluation**: ROC-AUC as primary metric, avoiding accuracy pitfalls
- ✅ **Model Persistence**: Save/load trained models for reproducibility
- ✅ **Bonus Features**: 3-class prediction, Self-Organizing Maps (SOM) visualization

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (tested on 3.13)
- Windows/Linux/macOS
- 8GB+ RAM recommended
- 5GB free disk space

### Installation & Setup

**Option 1: Automated Setup (Recommended)**
```batch
# Run the setup script
setup.bat
```

**Option 2: Manual Setup**
```batch
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

**Start Jupyter Notebook Interface:**
```batch
run.bat
```

**Run Specific Notebook:**
```batch
run_notebook.bat 1  # Step 1: Data Analysis
run_notebook.bat 2  # Step 2: Preprocessing
```

**Manual Execution:**
```batch
# Activate environment
.venv\Scripts\activate

# Start Jupyter
jupyter notebook
```

---

## 🗂️ Project Structure

```
ZOIDBERG2.0/
│
├── config/
│   └── config.yaml              # Centralized configuration
│
├── data/
│   ├── raw/
│   │   ├── dataset1/            # Training dataset 1
│   │   ├── dataset2/            # Training dataset 2
│   │   └── dataset3_tuning/     # Reserved for hyperparameter tuning
│   ├── interim/                 # Intermediate processed data
│   └── processed/               # Final processed data
│
├── notebooks/
│   ├── 01_Data_Analysis_Integrity_Check.ipynb
│   ├── 02_Preprocessing_FeatureEngineering.ipynb
│   ├── 03_Baseline_Modeling.ipynb
│   ├── 04_Deep_Learning_CNN.ipynb
│   ├── 05_Evaluation_ROC_AUC.ipynb
│   └── 06_Visualization_SOM.ipynb
│
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Dataset loading utilities
│   │   └── data_validator.py    # Data integrity checks
│   ├── features/
│   │   ├── preprocessing.py     # Image preprocessing
│   │   └── feature_extraction.py # PCA and feature engineering
│   ├── models/
│   │   ├── baseline_models.py   # Traditional ML models
│   │   ├── deep_learning.py     # CNN architectures
│   │   └── model_selection.py   # Hyperparameter tuning
│   ├── visualization/
│   │   ├── plots.py             # Plotting utilities
│   │   └── som_viz.py           # Self-Organizing Map visualization
│   └── utils/
│       └── helpers.py           # General utilities
│
├── models/
│   ├── saved_models/            # Trained models (.pkl, .h5)
│   └── checkpoints/             # Training checkpoints
│
├── reports/
│   ├── figures/                 # Generated plots and visualizations
│   ├── metrics/                 # Evaluation metrics
│   └── final_report.pdf         # Final synthesis document
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📚 Development Roadmap

### Step 1: Data Analysis & Integrity Check ✅ COMPLETE
**Notebook**: `01_Data_Analysis_Integrity_Check.ipynb`

**Objectives**:
- Load and inspect the 3 datasets
- Perform data integrity checks (corrupted images, format validation)
- Exploratory Data Analysis (EDA): class distribution, image dimensions, pixel statistics
- Dataset comparison and quality report generation

**Deliverables**:
- ✓ Data quality report (5,856 total images analyzed)
- ✓ Class distribution visualizations (2.89:1 imbalance ratio)
- ✓ Sample image visualizations (0 corrupted images found)

**Status**: Completed and validated

---

### Step 2: Preprocessing & Feature Engineering ✅ COMPLETE
**Notebook**: `02_Preprocessing_FeatureEngineering.ipynb`

**Objectives**:
- Implement image preprocessing pipeline (resizing, normalization)
- Data augmentation strategy
- Feature extraction from images
- PCA implementation for dimensionality reduction
- Setup train-test split and cross-validation folds

**Deliverables**:
- ✓ Preprocessed datasets ready for modeling (224x224 grayscale)
- ✓ PCA-transformed features (95% variance retained)
- ✓ Train-test split (80/20 stratified)
- ✓ 5-fold cross-validation setup

**Status**: Completed and ready for modeling

---

### Step 3: Baseline Modeling & Cross-Validation
**Notebook**: `03_Baseline_Modeling.ipynb`

**Objectives**:
- Implement baseline models (Logistic Regression, SVM, Random Forest, Gradient Boosting)
- Compare train-test split vs cross-validation results
- Initial model evaluation using ROC-AUC
- Feature importance analysis

**Deliverables**:
- Trained baseline models
- Performance comparison report
- Model persistence (saved models)

---

### Step 4: Deep Learning Implementation
**Notebook**: `04_Deep_Learning_CNN.ipynb`

**Objectives**:
- Implement CNN architecture (custom and pre-trained: VGG16, ResNet)
- Train deep learning models
- Hyperparameter tuning using Dataset3
- Implement 3-class classification (NORMAL, VIRUS, BACTERIA) - **Bonus**

**Deliverables**:
- Trained CNN models
- Training history plots
- Best model selection

---

### Step 5: Evaluation & ROC-AUC Analysis
**Notebook**: `05_Evaluation_ROC_AUC.ipynb`

**Objectives**:
- Comprehensive model evaluation using ROC-AUC
- Explain advantages of ROC-AUC over accuracy
- Generate ROC curves, confusion matrices
- Compare all models (baseline vs deep learning)
- Final model selection

**Deliverables**:
- ROC curves for all models
- Confusion matrices
- Comprehensive evaluation report

---

### Step 6: Visualization & SOM
**Notebook**: `06_Visualization_SOM.ipynb`

**Objectives**:
- Implement Self-Organizing Map (SOM) for feature visualization - **Bonus**
- Visualize learned representations
- Generate final visualizations for report
- Create synthesis document

**Deliverables**:
- SOM visualizations
- Final project report (PDF)


## 💻 Scripts & Usage

### Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.bat` | Initial setup & install dependencies | `setup.bat` |
| `run.bat` | Start Jupyter Notebook interface | `run.bat` |
| `run_notebook.bat` | Run specific notebook | `run_notebook.bat [1-2]` |

### Usage Examples

```batch
# First time setup
setup.bat

# Start project (opens Jupyter interface)
run.bat

# Run specific notebook
run_notebook.bat 1  # Data Analysis
run_notebook.bat 2  # Preprocessing
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ (tested on Python 3.13)
- pip package manager
- 8GB+ RAM recommended
- (Optional) GPU for deep learning training

### Installation

1. **Navigate to project directory**:
```bash
cd "T-DEV-810 -Zoidberg2.0-PAR_19"
```

2. **Run automated setup**:
```bash
setup.bat
```

OR **Manual setup**:
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Dataset already configured**:
- ✓ Datasets already in place:
  - `data/raw/dataset1/` (5,216 images)
  - `data/raw/dataset2/` (16 images)
  - `data/raw/dataset3_tuning/` (624 images)

### Configuration

Edit `config/config.yaml` to customize:
- Image processing parameters
- Model hyperparameters
- Training configuration
- File paths

---

## 📊 Usage

### Running Notebooks

Start with the first notebook and progress sequentially:

```bash
jupyter notebook notebooks/01_Data_Analysis_Integrity_Check.ipynb
```

Each notebook is self-contained and includes:
- Clear objectives
- Detailed explanations
- Executable code cells
- Visualizations
- Next steps

### Using Python Modules

You can also use the source modules directly:

```python
from src.data import DatasetLoader
from src.utils import set_seed

# Set reproducibility
set_seed(42)

# Load dataset
loader = DatasetLoader('config/config.yaml')
images, labels, class_names = loader.load_dataset('dataset1')
```

---

## 🎯 Key Design Decisions

### 1. Dataset Strategy
- **Dataset 1 & 2**: Used for training and validation
- **Dataset 3**: Reserved exclusively for hyperparameter tuning
- This prevents data leakage and ensures unbiased model selection

### 2. Evaluation Metric: ROC-AUC
**Why ROC-AUC over Accuracy?**
- Medical datasets are often imbalanced
- ROC-AUC is threshold-independent
- Provides insight into model's discriminative ability
- Better for comparing models in clinical settings

### 3. Cross-Validation vs Train-Test Split
- **Cross-Validation**: Better estimates of model performance, uses all data
- **Train-Test Split**: Faster, simpler, closer to production scenario
- We compare both to understand bias-variance tradeoffs

### 4. PCA Implementation
- Reduces dimensionality and computational cost
- Helps prevent overfitting
- Visualizes most important feature directions

---

## 📈 Model Persistence

All trained models are saved for reproducibility:

```python
from src.utils import save_model, load_model

# Save model
save_model(model, 'models/saved_models/cnn_model.pkl', 
          metadata={'accuracy': 0.95, 'roc_auc': 0.97})

# Load model
model, metadata = load_model('models/saved_models/cnn_model.pkl')
```

---

## 🔬 Research & Medical Context

**Pneumonia Detection**: Pneumonia is a serious lung infection that requires timely diagnosis. Chest X-rays are the primary diagnostic tool, but interpretation can be challenging and time-consuming. This project aims to assist radiologists with AI-powered screening.

**Clinical Relevance**:
- Early detection improves patient outcomes
- Reduces diagnostic time and workload
- Provides second opinion for challenging cases
- Can distinguish between viral and bacterial pneumonia (3-class model)


## 🤝 Contributing

This project follows industry best practices:
- **Code Style**: PEP 8
- **Documentation**: Comprehensive docstrings
- **Version Control**: Git with meaningful commits
- **Reproducibility**: Random seeds and saved models

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

**ML Architecture Team**  
EPITECH - T-DEV-810  
Medical Imaging / Computer Aided Diagnosis

---

## 🙏 Acknowledgments

- Medical imaging datasets providers
- Open-source ML community
- Scikit-learn, TensorFlow, and PyTorch teams

---

## 📧 Contact

For questions or collaboration:
- Project Repository: ZOIDBERG2.0
- Email: [Your email]

---

**Note**: This project is for educational and research purposes. Any clinical application requires rigorous validation and regulatory approval.
