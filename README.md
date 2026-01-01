# 🍅 Tomato Ripening Classification and Physical Parameter Estimation

> **Computer Vision System with Machine Learning for Non-Destructive Tomato Quality Assessment**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![INPI](https://img.shields.io/badge/INPI-Registration%20Pending-yellow.svg)](#-software-registration)

---

## 🎯 Overview

Computer vision system for **automatic classification** of tomato ripening stages (Green, Orange, Red) and **estimation of physical parameters** (mass, volume) using RGB images and classical ML.

**Key Results**: Random Forest achieved **97.14% accuracy** | Lasso R² = **0.72** for mass estimation

---

## 📁 Repository Structure
```
tomato-classification/
├── 📁 modelos/                              # Trained ML models (.pkl)
│   ├── modelo_classificacao_rf.pkl
│   ├── modelo_regressao_peso.pkl
│   ├── modelo_regressao_volume.pkl
│   ├── scaler_classificacao.pkl
│   ├── scaler_regressao.pkl
│   └── label_encoder.pkl
├── 📁 imagens/                              # Tomato sample images
├── 🐍 app_foto_3.py                         # Streamlit inference application
├── 📓 notebook_dissertacao_PUBLICAVEL.ipynb # Complete analysis notebook
├── 📊 oficial_experimento_tomates_2025.xlsx # Laboratory measurements
├── 📊 dataset_features_imagens.csv          # Extracted image features (66 samples)
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 LICENSE
```

---

## 📜 Software Registration

⚠️ **This software is under registration process at INPI (National Institute of Industrial Property - Brazil).**

---

## 🚀 Installation

### Local Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tomato-classification.git
cd tomato-classification

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app_foto_3.py
# Access: http://localhost:8501
```

### Google Colab

Open `notebook_dissertacao_PUBLICAVEL.ipynb` directly in Colab - no local installation required.

---

## 💻 Usage

### Streamlit App Features

- 📸 **Upload Tab**: Analyze images from file
- 🎥 **Webcam Tab**: Real-time capture and analysis
- ⚙️ **Sidebar**: Adjust pixels/cm scale

### Jupyter Notebook

The `notebook_dissertacao_PUBLICAVEL.ipynb` contains:
- Statistical analysis (ANOVA, Tukey HSD)
- Feature extraction pipeline
- Model training and evaluation
- Results visualization

---

## 🔬 Methodology

### Feature Extraction (22 features)

| Category | Features |
|----------|----------|
| **Color (RGB)** | R/G/B_mean, R/G/B_ratio, RG_ratio, RG_diff_norm |
| **Color (HSV)** | H_mean, S_mean, V_mean |
| **Color (CIELAB)** | L*, a*, b* |
| **Texture (GLCM)** | Contrast, Homogeneity, Energy, Correlation |
| **Geometric** | Area, Perimeter, Equivalent Diameter, Circularity |

### Models

- **Classification**: Random Forest, SVM (RBF), KNN (k=5)
- **Regression**: Lasso, Ridge (geometric features only)
- **Validation**: Stratified k-fold (k=10)

---

## 📊 Results

### Classification (k=10 cross-validation)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Random Forest** | **97.14%** | 0.9696 |
| SVM (RBF) | 96.90% | 0.9696 |
| KNN (k=5) | 95.48% | 0.9545 |

### Regression

| Target | Model | R² |
|--------|-------|-----|
| Mass | Lasso | 0.7229 |
| Volume | Ridge | 0.6455 |

---

## 📚 References

1. **Bello et al.** (2020). Digital image analysis for tomato quality assessment.
2. **Phan et al.** (2023). YOLOv5-based CNN for tomato classification.
3. **Costa et al.** (2025). Embedded computer vision for tomatoes.
4. **Giovannoni, J.** (2004). Genetic regulation of fruit ripening.

---

## 📝 Citation
```bibtex
@mastersthesis{author2025tomato,
  title  = {Characterization of Tomatoes by RGB Images and Machine Learning},
  school = {Graduate Program in Agrifood Technology (PPGTA)},
  year   = {2025}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

---

<p align="center">
  <b>PPGTA</b> • Master's Dissertation • 2025
</p>
