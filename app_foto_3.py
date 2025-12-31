# =============================================================================
# APLICAÇÃO STREAMLIT V3 LITE - CLASSIFICAÇÃO DE TOMATES
# Versão sem streamlit-webrtc (mais compatível)
# Usa st.camera_input para captura de imagens
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
from PIL import Image
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="🍅 Classificação de Tomates v3",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS CUSTOMIZADO
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 50%, #ffa502 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .resultado-grande {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
    }

    .verde { background: linear-gradient(135deg, #00d26a22, #00d26a44); color: #00d26a; border: 3px solid #00d26a; }
    .alaranjado { background: linear-gradient(135deg, #ffa50222, #ffa50244); color: #ffa502; border: 3px solid #ffa502; }
    .vermelho { background: linear-gradient(135deg, #e9456022, #e9456044); color: #e94560; border: 3px solid #e94560; }

    .info-box {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================

MODELO_DIR = 'modelos'

FEATURES_CLASSIFICACAO = [
    'area_cm2', 'perimetro_cm', 'diametro_equiv_cm', 'circularidade_img',
    'R_mean', 'G_mean', 'B_mean', 'R_ratio', 'G_ratio', 'B_ratio',
    'RG_ratio', 'RG_diff_norm',
    'H_mean', 'S_mean', 'V_mean',
    'L_mean', 'a_mean', 'b_mean',
    'GLCM_contrast', 'GLCM_homogeneity', 'GLCM_energy', 'GLCM_correlation'
]

FEATURES_REGRESSAO = ['area_cm2', 'perimetro_cm', 'diametro_equiv_cm', 'circularidade_img']


# =============================================================================
# FUNÇÕES DE LOG PARA TERMINAL
# =============================================================================

def log_inicializacao():
    """Log de inicialização do sistema."""
    print("\n" + "=" * 80)
    print("SISTEMA DE CLASSIFICAÇÃO DE TOMATES - PPGTA")
    print("Dissertação: Caracterização de Tomates por Imagens RGB e Aprendizado de Máquina")
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sistema inicializado")
    print("-" * 80)


def log_modelos_carregados():
    """Log de carregamento dos modelos."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Modelos carregados com sucesso:")
    print(f"    - Classificador: Random Forest (maturação)")
    print(f"    - Regressor Peso: Ridge Regression")
    print(f"    - Regressor Volume: Ridge Regression")
    print("-" * 80)


def log_features_extraidas(features_clf, features_reg, fonte):
    """Log das features extraídas da imagem."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EXTRAÇÃO DE FEATURES ({fonte})")
    print("-" * 40)
    print("Features Geométricas:")
    print(f"    Área (cm²):           {features_reg[0, 0]:.4f}")
    print(f"    Perímetro (cm):       {features_reg[0, 1]:.4f}")
    print(f"    Diâmetro Equiv (cm):  {features_reg[0, 2]:.4f}")
    print(f"    Circularidade:        {features_reg[0, 3]:.4f}")
    print("\nFeatures de Cor (RGB):")
    print(f"    R_mean: {features_clf[0, 4]:.2f} | G_mean: {features_clf[0, 5]:.2f} | B_mean: {features_clf[0, 6]:.2f}")
    print(
        f"    R_ratio: {features_clf[0, 7]:.4f} | G_ratio: {features_clf[0, 8]:.4f} | B_ratio: {features_clf[0, 9]:.4f}")
    print(f"    RG_ratio: {features_clf[0, 10]:.4f} | RG_diff_norm: {features_clf[0, 11]:.4f}")
    print("\nFeatures de Cor (HSV):")
    print(
        f"    H_mean: {features_clf[0, 12]:.2f} | S_mean: {features_clf[0, 13]:.2f} | V_mean: {features_clf[0, 14]:.2f}")
    print("\nFeatures de Cor (L*a*b*):")
    print(f"    L*: {features_clf[0, 15]:.2f} | a*: {features_clf[0, 16]:.2f} | b*: {features_clf[0, 17]:.2f}")
    print("\nFeatures de Textura (GLCM):")
    print(f"    Contrast: {features_clf[0, 18]:.4f} | Homogeneity: {features_clf[0, 19]:.4f}")
    print(f"    Energy: {features_clf[0, 20]:.4f} | Correlation: {features_clf[0, 21]:.4f}")


def log_resultado_predicao(resultado, fonte):
    """Log do resultado da predição."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RESULTADO DA PREDIÇÃO ({fonte})")
    print("=" * 40)
    print(f"    CLASSE: {resultado['classe'].upper()}")

    if resultado['probabilidades'] is not None:
        print("\n    Probabilidades por classe:")
        classes = ['alaranjado', 'verde', 'vermelho']
        for i, c in enumerate(classes):
            prob = resultado['probabilidades'][i] * 100
            print(f"        {c.capitalize():12s}: {prob:6.2f}%")

    print(f"\n    Peso estimado:   {resultado['peso']:.2f} g")
    print(f"    Volume estimado: {resultado['volume'] / 1000:.2f} cm³")
    print("=" * 40)
    print()


def log_erro_segmentacao(fonte):
    """Log quando a segmentação falha."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERRO: Tomate não detectado ({fonte})")
    print("    Verifique iluminação e posicionamento da amostra.")
    print("-" * 40)


# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================

@st.cache_resource
def carregar_modelos():
    try:
        modelos = {
            'scaler_clf': joblib.load(os.path.join(MODELO_DIR, 'scaler_classificacao.pkl')),
            'scaler_reg': joblib.load(os.path.join(MODELO_DIR, 'scaler_regressao.pkl')),
            'label_encoder': joblib.load(os.path.join(MODELO_DIR, 'label_encoder.pkl')),
            'modelo_clf': joblib.load(os.path.join(MODELO_DIR, 'modelo_classificacao_rf.pkl')),
            'modelo_peso': joblib.load(os.path.join(MODELO_DIR, 'modelo_regressao_peso.pkl')),
            'modelo_volume': joblib.load(os.path.join(MODELO_DIR, 'modelo_regressao_volume.pkl')),
            'sucesso': True
        }
        log_modelos_carregados()
        return modelos
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERRO ao carregar modelos: {str(e)}")
        return {'sucesso': False, 'erro': str(e)}


def segmentar_tomate(imagem_rgb):
    hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask3 = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))
    mask4 = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

    mask = mask1 | mask2 | mask3 | mask4

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    maior_contorno = max(contours, key=cv2.contourArea)

    if cv2.contourArea(maior_contorno) < 1000:
        return None, None

    mask_final = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_final, [maior_contorno], -1, 255, -1)

    return mask_final, maior_contorno


def extrair_features(imagem_rgb, pixels_por_cm):
    mask, contorno = segmentar_tomate(imagem_rgb)

    if mask is None:
        return None, None, None

    area_px = cv2.contourArea(contorno)
    perimetro_px = cv2.arcLength(contorno, True)
    area_cm2 = area_px / (pixels_por_cm ** 2)
    perimetro_cm = perimetro_px / pixels_por_cm
    diametro_equiv_cm = np.sqrt(4 * area_cm2 / np.pi)
    circularidade = (4 * np.pi * area_px) / (perimetro_px ** 2) if perimetro_px > 0 else 0

    pixels = imagem_rgb[mask > 0]
    if len(pixels) == 0:
        return None, None, None

    R_mean, G_mean, B_mean = pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean()
    total = R_mean + G_mean + B_mean
    R_ratio = R_mean / total if total > 0 else 0.33
    G_ratio = G_mean / total if total > 0 else 0.33
    B_ratio = B_mean / total if total > 0 else 0.33
    RG_ratio = R_mean / G_mean if G_mean > 0 else 1
    RG_diff_norm = (R_mean - G_mean) / 255.0

    hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    pixels_hsv = hsv[mask > 0]
    H_mean, S_mean, V_mean = pixels_hsv[:, 0].mean(), pixels_hsv[:, 1].mean(), pixels_hsv[:, 2].mean()

    lab = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2LAB)
    pixels_lab = lab[mask > 0]
    L_mean = pixels_lab[:, 0].mean()
    a_mean = pixels_lab[:, 1].mean() - 128
    b_mean_lab = pixels_lab[:, 2].mean() - 128

    gray = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)
    pixels_gray = gray[mask > 0]
    variance = pixels_gray.var() if len(pixels_gray) > 0 else 0
    hist, _ = np.histogram(pixels_gray, bins=256, range=(0, 256), density=True)

    features_clf = np.array([[
        area_cm2, perimetro_cm, diametro_equiv_cm, circularidade,
        R_mean, G_mean, B_mean, R_ratio, G_ratio, B_ratio, RG_ratio, RG_diff_norm,
        H_mean, S_mean, V_mean, L_mean, a_mean, b_mean_lab,
        variance / 100, 1 / (1 + variance / 1000), np.sum(hist ** 2), 0.97
    ]])

    features_reg = np.array([[area_cm2, perimetro_cm, diametro_equiv_cm, circularidade]])

    return features_clf, features_reg, mask


def realizar_predicao(features_clf, features_reg, modelos):
    features_clf_scaled = modelos['scaler_clf'].transform(features_clf)
    features_reg_scaled = modelos['scaler_reg'].transform(features_reg)

    classe_idx = modelos['modelo_clf'].predict(features_clf_scaled)[0]
    classe = modelos['label_encoder'].inverse_transform([classe_idx])[0]

    probs = None
    if hasattr(modelos['modelo_clf'], 'predict_proba'):
        probs = modelos['modelo_clf'].predict_proba(features_clf_scaled)[0]

    return {
        'classe': classe,
        'classe_idx': classe_idx,
        'probabilidades': probs,
        'peso': modelos['modelo_peso'].predict(features_reg_scaled)[0],
        'volume': modelos['modelo_volume'].predict(features_reg_scaled)[0]
    }


def criar_grafico_resultado(resultado, modelos):
    """Gráfico com layout corrigido - peso e volume separados."""
    cores = {'verde': '#00d26a', 'alaranjado': '#ffa502', 'vermelho': '#e94560'}
    cor_classe = cores.get(resultado['classe'], '#888')

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "bar"}],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=("Classificação", "Probabilidades", "Peso Estimado", "Volume Estimado"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    confianca = resultado['probabilidades'][resultado['classe_idx']] * 100 if resultado[
                                                                                  'probabilidades'] is not None else 100

    # Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confianca,
            title={'text': f"<b>{resultado['classe'].upper()}</b>", 'font': {'size': 22, 'color': cor_classe}},
            number={'suffix': '%', 'font': {'size': 32, 'color': cor_classe}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': cor_classe, 'thickness': 0.8},
                'bgcolor': 'rgba(128,128,128,0.2)',
                'borderwidth': 2,
                'bordercolor': cor_classe,
            }
        ),
        row=1, col=1
    )

    # Barras
    if resultado['probabilidades'] is not None:
        classes = modelos['label_encoder'].classes_
        probs = resultado['probabilidades'] * 100
        fig.add_trace(
            go.Bar(
                x=[c.capitalize() for c in classes],
                y=probs,
                marker_color=[cores.get(c, '#888') for c in classes],
                text=[f'{p:.1f}%' for p in probs],
                textposition='outside',
                textfont={'size': 12}
            ),
            row=1, col=2
        )

    # PESO
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=resultado['peso'],
            title={'text': "PESO", 'font': {'size': 16}},
            number={'suffix': ' g', 'font': {'size': 42, 'color': '#4ecdc4'}}
        ),
        row=2, col=1
    )

    # VOLUME
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=resultado['volume'] / 1000,
            title={'text': "VOLUME", 'font': {'size': 16}},
            number={'suffix': ' cm³', 'font': {'size': 42, 'color': '#ff6b6b'}, 'valueformat': '.1f'}
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig.update_yaxes(range=[0, 110], row=1, col=2)

    return fig


# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

def main():
    # Log de inicialização (apenas uma vez)
    if 'initialized' not in st.session_state:
        log_inicializacao()
        st.session_state.initialized = True

    st.markdown('<p class="main-header">🍅 Sistema de Classificação de Tomates</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visão Computacional + ML | PPGTA v3 Lite</p>', unsafe_allow_html=True)

    modelos = carregar_modelos()

    if not modelos.get('sucesso', False):
        st.error("❌ Erro ao carregar modelos")
        return

    st.success("✅ Modelos carregados!")

    # Sidebar
    st.sidebar.markdown("## ⚙️ Configurações")
    pixels_por_cm = st.sidebar.slider("Pixels por cm", 20.0, 100.0, 20.0, 0.1)

    # Log da configuração de escala
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Escala configurada: {pixels_por_cm} pixels/cm")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Upload", "🎥 Webcam", "ℹ️ Sobre"])

    # TAB 1: Upload
    with tab1:
        st.markdown("### 📸 Análise por Upload")

        uploaded_file = st.file_uploader("Faça upload de uma imagem", type=['jpg', 'jpeg', 'png', 'bmp'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            imagem_rgb = np.array(image.convert('RGB'))

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📷 Imagem Original")
                st.image(imagem_rgb, use_container_width=True)

            features_clf, features_reg, mask = extrair_features(imagem_rgb, pixels_por_cm)

            if mask is not None:
                with col2:
                    st.markdown("#### 🎯 Segmentação")
                    img_seg = imagem_rgb.copy()
                    img_seg[mask == 0] = [240, 240, 240]
                    st.image(img_seg, use_container_width=True)

                # Log das features extraídas
                log_features_extraidas(features_clf, features_reg, f"Upload: {uploaded_file.name}")

                resultado = realizar_predicao(features_clf, features_reg, modelos)

                # Log do resultado
                log_resultado_predicao(resultado, f"Upload: {uploaded_file.name}")

                st.markdown("---")
                st.markdown("### 📊 Resultado da Análise")
                fig = criar_grafico_resultado(resultado, modelos)
                st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.warning("⚠️ Tomate não detectado")
                log_erro_segmentacao(f"Upload: {uploaded_file.name}")

    # TAB 2: Webcam
    with tab2:
        st.markdown("### 🎥 Análise via Webcam")

        camera_image = st.camera_input("📷 Capturar imagem")

        if camera_image is not None:
            image = Image.open(camera_image)
            imagem_rgb = np.array(image.convert('RGB'))

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📷 Imagem Capturada")
                st.image(imagem_rgb, use_container_width=True)

            features_clf, features_reg, mask = extrair_features(imagem_rgb, pixels_por_cm)

            if mask is not None:
                with col2:
                    st.markdown("#### 🎯 Segmentação")
                    img_seg = imagem_rgb.copy()
                    img_seg[mask == 0] = [240, 240, 240]
                    st.image(img_seg, use_container_width=True)

                # Log das features extraídas
                log_features_extraidas(features_clf, features_reg, "Webcam")

                resultado = realizar_predicao(features_clf, features_reg, modelos)

                # Log do resultado
                log_resultado_predicao(resultado, "Webcam")

                # Resultado em destaque
                emoji = {'verde': '🟢', 'alaranjado': '🟠', 'vermelho': '🔴'}
                st.markdown(f"""
                <div class="resultado-grande {resultado['classe']}">
                    {emoji[resultado['classe']]} {resultado['classe'].upper()}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📊 Resultado da Análise")
                fig = criar_grafico_resultado(resultado, modelos)
                st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.warning("⚠️ Tomate não detectado")
                log_erro_segmentacao("Webcam")

    # TAB 3: Sobre
    with tab3:
        st.markdown("### ℹ️ Sobre o Sistema")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🎯 Objetivo
            Classificação não-destrutiva de tomates.

            #### 📊 Dataset
            - 66 amostras (22/classe)
            - Verde, Alaranjado, Vermelho
            """)

        with col2:
            st.markdown("""
            #### 🆕 Correções v3 Lite
            - ✅ Layout corrigido (peso/volume)
            - ✅ Logs para documentação
            - ✅ Sem dependências extras

            #### 👨‍🎓 PPGTA
            """)


if __name__ == "__main__":
    main()
