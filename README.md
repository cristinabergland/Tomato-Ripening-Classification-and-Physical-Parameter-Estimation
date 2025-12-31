# 🍅 Tomato Ripening Classification and Physical Parameter Estimation

> **Computer Vision System with Machine Learning for Non-Destructive Tomato Quality Assessment**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow.svg)](https://colab.research.google.com/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Statistical Analysis](#1-statistical-analysis)
  - [Machine Learning Training](#2-machine-learning-training)
  - [Streamlit Inference App](#3-streamlit-inference-app)
- [Methodology](#-methodology)
- [Results](#-results)
- [Dataset](#-dataset)
- [References](#-references)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This repository contains the complete implementation of a **computer vision system integrated with classical machine learning algorithms** for:

1. **Automatic classification** of tomato ripening stages (Green, Orange, Red)
2. **Estimation of physical parameters** (mass and volume) from RGB images

The system was developed as part of a Master's dissertation in the **Graduate Program in Agrifood Technology (PPGTA)**, focusing on the **Santa Cruz Kada** tomato cultivar.

### Why Classical ML over Deep Learning?

| Aspect | Classical ML (This Work) | Deep Learning |
|--------|-------------------------|---------------|
| Dataset Size | ✅ Works with n=66 | ❌ Requires thousands |
| Computational Cost | ✅ Low (CPU-only) | ❌ High (GPU required) |
| Interpretability | ✅ Feature importance | ❌ Black box |
| Deployment | ✅ Edge devices | ❌ Server infrastructure |
| Accuracy | 97.14% | ~97.3% (Phan et al., 2023) |

---

## ✨ Key Features

- 📊 **Complete statistical analysis pipeline**: ANOVA, Tukey HSD, correlation matrices
- 🤖 **Machine Learning models**: Random Forest, SVM, KNN for classification; Lasso, Ridge for regression
- 📸 **Feature extraction**: RGB, HSV, CIELAB color spaces + GLCM texture descriptors
- 🔬 **Metrological validation**: Bland-Altman analysis for colorimetric comparison
- 🖥️ **Interactive Streamlit app**: Real-time inference via upload or webcam
- 📈 **Publication-ready outputs**: High-resolution figures and formatted tables

---

## 📁 Project Structure

```
tomato-classification/
│
├── 📓 notebooks/
│   ├── 01_statistical_analysis.ipynb      # ANOVA, Tukey, descriptive stats
│   ├── 02_feature_extraction.ipynb        # Image processing pipeline
│   ├── 03_classification_models.ipynb     # RF, SVM, KNN training
│   ├── 04_regression_models.ipynb         # Lasso, Ridge for mass/volume
│   └── 05_model_evaluation.ipynb          # Learning curves, validation
│
├── 📱 app/
│   ├── app_foto_3.py                      # Streamlit inference application
│   └── modelos/                           # Trained model files (.pkl)
│       ├── modelo_classificacao_rf.pkl
│       ├── modelo_regressao_peso.pkl
│       ├── modelo_regressao_volume.pkl
│       ├── scaler_classificacao.pkl
│       ├── scaler_regressao.pkl
│       └── label_encoder.pkl
│
├── 📊 data/
│   ├── oficial_experimento_tomates_2025.xlsx   # Laboratory measurements
│   └── dataset_features_imagens.csv            # Extracted image features
│
├── 📈 results/
│   ├── tabelas/                           # CSV output tables
│   └── figuras/                           # PNG/PDF figures
│
├── 📚 references/                         # Scientific papers (PDF)
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── LICENSE                                # MIT License
```

---

## 🚀 Installation

### Option 1: Google Colab (Recommended)

Open notebooks directly in Colab - no local installation required:

```python
# Run this cell in Colab to install dependencies
!pip install pandas numpy scikit-learn matplotlib seaborn plotly opencv-python-headless scipy statsmodels joblib
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tomato-classification.git
cd tomato-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
opencv-python>=4.5.0
scipy>=1.7.0
statsmodels>=0.13.0
joblib>=1.1.0
streamlit>=1.28.0
Pillow>=8.0.0
openpyxl>=3.0.0
```

---

## 💻 Usage

### 1. Statistical Analysis

```python
# Run in Jupyter/Colab
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Load data
df = pd.read_excel('data/oficial_experimento_tomates_2025.xlsx')

# ANOVA One-Way
verde = df[df['tratamento'] == 'verde']['a_lab']
alaranjado = df[df['tratamento'] == 'alaranjado']['a_lab']
vermelho = df[df['tratamento'] == 'vermelho']['a_lab']

f_stat, p_value = f_oneway(verde, alaranjado, vermelho)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.4e}")

# Tukey HSD Test
tukey = pairwise_tukeyhsd(df['a_lab'], df['tratamento'], alpha=0.05)
print(tukey.summary())
```

### 2. Machine Learning Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Prepare data
X = df[FEATURES_CLASSIFICACAO]
y = df['tratamento']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)

# Cross-validation (k=10)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Save model
rf.fit(X_scaled, y_encoded)
joblib.dump(rf, 'modelos/modelo_classificacao_rf.pkl')
```

### 3. Streamlit Inference App

```bash
# Run the application
cd app
streamlit run app_foto_3.py

# Access in browser: http://localhost:8501
```

**App Features:**
- 📸 **Upload Tab**: Analyze images from file
- 🎥 **Webcam Tab**: Real-time capture and analysis
- ⚙️ **Sidebar**: Adjust pixels/cm scale for dimensional accuracy

---

## 🔬 Methodology

### Image Acquisition Protocol

| Parameter | Specification |
|-----------|---------------|
| Camera | RGB sensor, 12MP |
| Lighting | Standardized D65 white LED |
| Background | Matte black (non-reflective) |
| Distance | 30 cm (fixed) |
| Resolution | 20 pixels/cm (calibrated) |

### Feature Extraction Pipeline

```
RGB Image → Segmentation (HSV thresholding) → Feature Extraction
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
              Color Features              Geometric Features              Texture Features
              ├─ RGB (R,G,B mean/std)     ├─ Area (cm²)                  ├─ GLCM Contrast
              ├─ HSV (H,S,V mean/std)     ├─ Perimeter (cm)             ├─ GLCM Homogeneity
              ├─ CIELAB (L*,a*,b*)        ├─ Equivalent Diameter        ├─ GLCM Energy
              └─ Ratios (R/G, etc.)       └─ Circularity                └─ GLCM Correlation
```

### Classification Features (22 total)

| Category | Features | Purpose |
|----------|----------|---------|
| **Colorimetric** | R/G/B_mean, R/G/B_ratio, RG_ratio, RG_diff_norm | Capture chlorophyll→carotenoid transition |
| **HSV** | H_mean, S_mean, V_mean | Perceptual color representation |
| **CIELAB** | L_mean, a_mean, b_mean | Device-independent color space |
| **Texture** | GLCM contrast, homogeneity, energy, correlation | Surface texture characterization |
| **Geometric** | area_cm2, perimetro_cm, diametro_equiv_cm, circularidade | Size normalization (not discriminative) |

### Regression Features (4 only)

For mass and volume estimation, **only geometric features** are used (area, perimeter, diameter, circularity), as color features have no physical correlation with these parameters.

---

## 📊 Results

### Classification Performance

| Model | Accuracy | Std | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| **Random Forest** | **97.14%** | 4.71% | 0.9722 | 0.9697 | 0.9696 |
| SVM (RBF) | 96.90% | 4.21% | 0.9722 | 0.9697 | 0.9696 |
| KNN (k=5) | 95.48% | 5.94% | 0.9551 | 0.9545 | 0.9545 |

### Confusion Matrix (Random Forest)

```
                 Predicted
              Verde  Alaranjado  Vermelho
Actual Verde    22       0          0
    Alaranjado   0      21          1
      Vermelho   0       1         21
```

### Regression Performance

| Target | Best Model | R² | RMSE | Features |
|--------|------------|-----|------|----------|
| **Mass** | Lasso | 0.7229 | 15.8 g | Geometric (4) |
| **Volume** | Ridge | 0.6455 | 12.3 cm³ | Geometric (4) |

> **Note**: R² values of 0.65-0.72 represent the methodological ceiling for monocular vision with aggregated geometric features, not modeling inadequacy.

### Comparison with Literature

| Study | Method | Classes | Accuracy |
|-------|--------|---------|----------|
| **This Study** | Random Forest | 3 | 97.14% |
| Bello et al. (2020) | k-means + RGB | 3 | 98.0% |
| Phan et al. (2023) | YOLOv5 (CNN) | 6 | 97.3% |
| Ningsih & Cholidhazia (2022) | KNN + RGB/HSV | 5 | 91.25% |

---

## 📁 Dataset

### Laboratory Measurements (`oficial_experimento_tomates_2025.xlsx`)

| Variable | Unit | Description |
|----------|------|-------------|
| peso | g | Fresh mass |
| volume | mm³ | Water displacement |
| brix | °Brix | Soluble solids content |
| ph | - | Acidity level |
| acidez_pct | % | Titratable acidity |
| dureza | N | Firmness (penetrometer) |
| L_lab, a_lab, b_lab | - | CIELAB colorimetry (reference) |

### Image Features (`dataset_features_imagens.csv`)

- **66 samples** (22 per ripening stage)
- **36 features** per sample
- **3 classes**: verde (green), alaranjado (orange), vermelho (red)

---

## 📚 References

Key scientific references supporting this work:

1. **Bello, F. et al.** (2020). Digital image analysis and colorimetric indices for tomato quality assessment. *Journal of Food Engineering*.

2. **Phan, Q. H. et al.** (2023). YOLOv5-based CNN for tomato ripening classification. *Computers and Electronics in Agriculture*.

3. **Ningsih, L. & Cholidhazia, P.** (2022). Classification of tomato maturity levels based on RGB and HSV colors using KNN algorithm. *Journal of Artificial Intelligence and Digital Business*.

4. **Costa, A. G. et al.** (2025). Embedded computer vision for physical parameter estimation in tomatoes. *Scientia Horticulturae*.

5. **Giovannoni, J.** (2004). Genetic regulation of fruit development and ripening. *The Plant Cell*.

6. **Breiman, L.** (2001). Random Forests. *Machine Learning*.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{author2025tomato,
  author       = {Author Name},
  title        = {Characterization of Tomatoes by RGB Images and Machine Learning: 
                  Classification of Ripening Stage and Estimation of Physical Parameters},
  school       = {Graduate Program in Agrifood Technology (PPGTA)},
  year         = {2025},
  type         = {Master's Dissertation}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Author**: [Your Name]  
**Email**: [your.email@institution.edu]  
**Institution**: Graduate Program in Agrifood Technology (PPGTA)

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love">
  <img src="https://img.shields.io/badge/Python-🐍-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Science-🔬-green.svg" alt="Science">
</p>
