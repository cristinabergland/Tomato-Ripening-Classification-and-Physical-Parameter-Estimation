# 🍅 Classificação de Maturação de Tomates e Estimativa de Parâmetros Físicos

> **Sistema de Visão Computacional com Aprendizado de Máquina para Avaliação Não-Destrutiva de Qualidade de Tomates**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![INPI](https://img.shields.io/badge/INPI-Registro%20em%20Andamento-yellow.svg)](#-registro-de-software)

---

## 🎯 Visão Geral

Sistema de visão computacional para **classificação automática** dos estádios de maturação de tomates (Verde, Alaranjado, Vermelho) e **estimativa de parâmetros físicos** (massa, volume) utilizando imagens RGB e aprendizado de máquina clássico.

**Principais Resultados**: Random Forest alcançou **97,14% de acurácia** | Lasso R² = **0,72** para estimativa de massa

---

## 📁 Estrutura do Repositório
```
tomato-classification/
├── 📁 modelos/                              # Modelos ML treinados (.pkl)
│   ├── modelo_classificacao_rf.pkl
│   ├── modelo_regressao_peso.pkl
│   ├── modelo_regressao_volume.pkl
│   ├── scaler_classificacao.pkl
│   ├── scaler_regressao.pkl
│   └── label_encoder.pkl
├── 📁 imagens_tomate.zip                    # Imagens das amostras de tomate
├── 🐍 app_foto_3.py                         # Aplicação Streamlit para inferência
├── 📓 notebook_dissertacao_PUBLICAVEL.ipynb # Notebook completo de análise
├── 📊 oficial_experimento_tomates_2025.xlsx # Medições laboratoriais
├── 📊 dataset_features_imagens.csv          # Features extraídas das imagens (66 amostras)
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 LICENSE
```

---

## 📜 Registro de Software

⚠️ **Este software está em processo de registro junto ao INPI (Instituto Nacional da Propriedade Industrial).**

---

## 🚀 Instalação

### Instalação Local
```bash
# Clonar repositório
git clone https://github.com/SEU_USUARIO/tomato-classification.git
cd tomato-classification

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação Streamlit
streamlit run app_foto_3.py
# Acesse: http://localhost:8501
```

### Google Colab

Abra `notebook_dissertacao_PUBLICAVEL.ipynb` diretamente no Colab - sem necessidade de instalação local.

---

## 💻 Uso

### Funcionalidades da Aplicação Streamlit

- 📸 **Aba Upload**: Analise imagens de arquivo
- 🎥 **Aba Webcam**: Captura e análise em tempo real
- ⚙️ **Barra Lateral**: Ajuste de escala pixels/cm

### Jupyter Notebook

O `notebook_dissertacao_PUBLICAVEL.ipynb` contém:
- Análise estatística (ANOVA, Tukey HSD)
- Pipeline de extração de features
- Treinamento e avaliação de modelos
- Visualização de resultados

---

## 🔬 Metodologia

### Extração de Features (22 características)

| Categoria | Features |
|-----------|----------|
| **Cor (RGB)** | R/G/B_mean, R/G/B_ratio, RG_ratio, RG_diff_norm |
| **Cor (HSV)** | H_mean, S_mean, V_mean |
| **Cor (CIELAB)** | L*, a*, b* |
| **Textura (GLCM)** | Contraste, Homogeneidade, Energia, Correlação |
| **Geométricas** | Área, Perímetro, Diâmetro Equivalente, Circularidade |

### Modelos

- **Classificação**: Random Forest, SVM (RBF), KNN (k=5)
- **Regressão**: Lasso, Ridge (apenas features geométricas)
- **Validação**: k-fold estratificado (k=10)

---

## 📊 Resultados

### Classificação (validação cruzada k=10)

| Modelo | Acurácia | F1-Score |
|--------|----------|----------|
| **Random Forest** | **97,14%** | 0,9696 |
| SVM (RBF) | 96,90% | 0,9696 |
| KNN (k=5) | 95,48% | 0,9545 |

### Regressão

| Alvo | Modelo | R² |
|------|--------|-----|
| Massa | Lasso | 0,7229 |
| Volume | Ridge | 0,6455 |

---

## 📚 Referências

1. **Bello et al.** (2020). Análise digital de imagens para avaliação de qualidade de tomates.
2. **Phan et al.** (2023). CNN baseada em YOLOv5 para classificação de tomates.
3. **Costa et al.** (2025). Visão computacional embarcada para tomates.
4. **Giovannoni, J.** (2004). Regulação genética do amadurecimento de frutos.

---

## 📝 Citação
```bibtex
@mastersthesis{autor2025tomate,
  title  = {Caracterização de Tomates por Imagens RGB e Aprendizado de Máquina},
  school = {Programa de Pós-Graduação em Tecnologia de Alimentos (PPGTA)},
  year   = {2025}
}
```

---

## 📄 Licença

Licença MIT - Veja o arquivo [LICENSE](LICENSE).

---

<p align="center">
  <b>PPGTA</b> • Dissertação de Mestrado • 2025
</p>
