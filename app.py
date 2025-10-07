import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# CONFIGURAÇÃO INICIAL
# =====================================
st.set_page_config(page_title="Predição e Dashboard de Obesidade", page_icon="🧠", layout="wide")

# =====================================
# MENU LATERAL
# =====================================
st.sidebar.title("📋 Navegação")
pagina = st.sidebar.radio("Ir para:", ["🏠 Predição", "📊 Dashboard"])

# =====================================
# CARREGAR MODELO E ENCODERS
# =====================================
modelo = joblib.load("C:/Users/phill/projetos/TECH4/knn_pipeline.pkl")
le_freq = joblib.load("C:/Users/phill/projetos/TECH4/encoder_frequencia.joblib")
le_comida = joblib.load("C:/Users/phill/projetos/TECH4/encoder_comida.joblib")
le_obesidade = joblib.load("C:/Users/phill/projetos/TECH4/encoder_obesidade.joblib")

# =====================================
# PÁGINA 1 – PREVISÃO
# =====================================
if pagina == "🏠 Predição":
    st.title("🧠 Predição de Obesidade - KNN Pipeline")
    st.markdown("Preencha os dados abaixo para prever o nível de obesidade.")

    genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
    idade = st.number_input("Idade", min_value=0, max_value=120, value=25)
    altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
    historico_familiar = st.selectbox("Histórico Familiar de Obesidade", ["Sim", "Não"])
    fumar = st.selectbox("Fuma?", ["Sim", "Não"])
    freq_alcool = st.selectbox("Frequência de Consumo de Álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
    comida_entre_refeicoes = st.selectbox("Come entre refeições?", ["Às vezes", "Frequentemente", "Sempre", "Não"])

    # Monta o DataFrame para o modelo
    entrada = pd.DataFrame({
        "Genero": [genero],
        "Idade": [idade],
        "Altura": [altura],
        "Peso": [peso],
        "Historico_Familiar": [historico_familiar],
        "Fumar": [fumar],
        "frequencia_consumo_alcool": [le_freq.transform([freq_alcool])[0]],
        "comida_entre_refeicoes": [le_comida.transform([comida_entre_refeicoes])[0]]
    })

    # Botão de previsão
    if st.button("🔍 Prever Obesidade"):
        try:
            predicao = modelo.predict(entrada)
            predicao_decodificada = le_obesidade.inverse_transform(predicao)
            st.success(f"**Resultado da previsão:** {predicao_decodificada[0]}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar a previsão: {e}")

    st.subheader("📋 Dados informados:")
    st.dataframe(entrada)

# =====================================
# PÁGINA 2 – DASHBOARD
# =====================================
elif pagina == "📊 Dashboard":
    st.title("📊 Dashboard Analítico - Obesidade")

    # Carregar dataset transformado
    df = pd.read_csv("C:/Users/phill/projetos/TECH4/dados_transformados.csv")

    # Barra lateral - filtros
    st.sidebar.subheader("Filtros")
    genero_filtro = st.sidebar.multiselect(
        "Filtrar por Gênero",
        df["Genero"].unique(),
        default=df["Genero"].unique()
    )

    # Aplicar filtros
    df_filtrado = df[df["Genero"].isin(genero_filtro)]

    # Criar layout de grade 2x2
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # === Gráfico 1 - Distribuição dos Níveis de Obesidade ===
    sns.countplot(
        data=df_filtrado,
        x="Obesidade",
        palette="Set2",
        ax=axs[0, 0]
    )
    axs[0, 0].set_title("Distribuição dos Níveis de Obesidade", fontsize=12, fontweight='bold')
    axs[0, 0].tick_params(axis='x', rotation=20)

    # === Gráfico 2 - Peso médio por tipo de obesidade ===
    media_peso = df_filtrado.groupby("Obesidade")["Peso"].mean().reset_index()
    sns.barplot(
        data=media_peso,
        x="Obesidade",
        y="Peso",
        palette="coolwarm",
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Peso Médio por Tipo de Obesidade", fontsize=12, fontweight='bold')
    axs[0, 1].tick_params(axis='x', rotation=20)

    # === Gráfico 3 - Influência do Gênero na Obesidade ===
    sns.countplot(
        data=df_filtrado,
        x="Obesidade",
        hue="Genero",
        palette="Set1",
        ax=axs[1, 0]
    )
    axs[1, 0].set_title("Influência do Gênero na Obesidade", fontsize=12, fontweight='bold')
    axs[1, 0].tick_params(axis='x', rotation=20)

    # === Gráfico 4 - Frequência de Consumo de Álcool x Obesidade ===
    sns.countplot(
        data=df_filtrado,
        x="frequencia_consumo_alcool",
        hue="Obesidade",
        palette="Set3",
        ax=axs[1, 1]
    )
    axs[1, 1].set_title("Consumo de Álcool x Tipo de Obesidade", fontsize=12, fontweight='bold')
    axs[1, 1].tick_params(axis='x', rotation=20)

    # Exibir todos os gráficos no Streamlit
    st.pyplot(fig)


