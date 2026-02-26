# ============================================================
# APP STREAMLIT - CREDIT SCORING EXECUTIVO
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from custom_transformers import Winsorizer

# ------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------

st.set_page_config(page_title="Credit Scoring", layout="wide")

st.title("💳 Credit Scoring - LightGBM")
st.write("Faça upload de um arquivo CSV para escoragem e análise de risco.")

# ------------------------------------------------------------
# CARREGANDO MODELO TREINADO
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model_final.pkl")

pipeline_modelo = load_model()

# ------------------------------------------------------------
# UPLOAD DO CSV
# ------------------------------------------------------------

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📂 Base carregada:")
    st.dataframe(df.head())

    try:
        # --------------------------------------------------------
        # ESCORAGEM
        # --------------------------------------------------------
        probabilidade = pipeline_modelo.predict_proba(df)[:, 1]
        df['score_probabilidade'] = probabilidade

        st.subheader("📊 Resultado da Escoragem:")
        st.dataframe(df.head())

        # --------------------------------------------------------
        # CLASSIFICAÇÃO POR PERCENTIL
        # --------------------------------------------------------

        st.header("📌 Classificação de Risco")

        st.write("""
        Como a base é desbalanceada, utilizamos classificação por percentil.
        Os 5% clientes com maior probabilidade estimada são classificados
        como Alto Risco.
        """)

        cutoff = df['score_probabilidade'].quantile(0.95)

        df['classe_risco'] = df['score_probabilidade'].apply(
            lambda x: "Alto Risco" if x >= cutoff else "Baixo Risco"
        )

        proporcao = df['classe_risco'].value_counts(normalize=True)

        col1, col2 = st.columns(2)

        col1.metric("🔴 Alto Risco (%)", f"{proporcao.get('Alto Risco',0):.2%}")
        col2.metric("🟢 Baixo Risco (%)", f"{proporcao.get('Baixo Risco',0):.2%}")

        # --------------------------------------------------------
        # FEATURE IMPORTANCE
        # --------------------------------------------------------

        st.header("📈 Importância das Variáveis")

        st.write("""
        O gráfico abaixo mostra quais variáveis mais contribuíram
        para as decisões do modelo LightGBM.
        A métrica utilizada é Gain, que mede o quanto cada variável
        reduziu o erro durante o treinamento.
        """)

        modelo = pipeline_modelo.named_steps['modelo']
        preprocessador = pipeline_modelo.named_steps['preprocessamento']

        nomes_features = []

        for nome, transformador, colunas in preprocessador.transformers_:
            if nome == 'num':
                nomes_features.extend(colunas)
            elif nome == 'dummy':
                encoder = transformador.named_steps['onehot']
                nomes_cat = encoder.get_feature_names_out(colunas)
                nomes_features.extend(nomes_cat)

        importancias_gain = modelo.booster_.feature_importance(importance_type='gain')

        df_importancia = pd.DataFrame({
            'variavel': nomes_features,
            'importancia': importancias_gain
        }).sort_values(by='importancia', ascending=False)

        fig1, ax1 = plt.subplots(figsize=(8,6))
        ax1.barh(df_importancia['variavel'][:15], df_importancia['importancia'][:15])
        ax1.invert_yaxis()
        ax1.set_title("Top 15 Variáveis Mais Importantes")
        st.pyplot(fig1)

        # --------------------------------------------------------
        # CURVA LIFT
        # --------------------------------------------------------

        st.header("📉 Curva Lift")

        st.write("""
        A Curva Lift avalia o poder de ranqueamento do modelo.
        Ela mostra quantas vezes o modelo concentra inadimplentes
        nos grupos de maior risco comparado à média geral.
        """)

        if 'mau' in df.columns:

            df_lift = df.sort_values(by='score_probabilidade', ascending=False).copy()
            df_lift['decil'] = pd.qcut(df_lift.index, 10, labels=False)

            taxa_media = df_lift['mau'].mean()

            lift_table = df_lift.groupby('decil')['mau'].agg(['count','sum'])
            lift_table['taxa_inadimplencia'] = lift_table['sum'] / lift_table['count']
            lift_table['lift'] = lift_table['taxa_inadimplencia'] / taxa_media

            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.plot(lift_table.index + 1, lift_table['lift'], marker='o')
            ax2.set_title("Curva Lift")
            ax2.set_xlabel("Decil (1 = Maior Risco)")
            ax2.set_ylabel("Lift")
            ax2.invert_xaxis()

            st.pyplot(fig2)

        else:
            st.info("Coluna 'mau' não encontrada. Curva Lift não pode ser calculada.")

        # --------------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------------

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Baixar base escorada",
            data=csv,
            file_name='base_escorada.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Erro ao aplicar modelo: {e}")