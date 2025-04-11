import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

st.set_page_config(
    page_title="Dashboard de Propensión a Seguros",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🔍 Dashboard de Propensión a Contratar Seguro")
st.markdown("Análisis de segmentos y características que influyen en la propensión a contratar seguros.")

# Cargar datos
df = pd.read_csv("data/processed/final_dataset.csv")
model = joblib.load("outputs/models/xgboost_tuned_model.pkl")
X = df.drop(columns=["has_insurance"])
df["pred_proba"] = model.predict_proba(X)[:, 1]
df["segment"] = pd.cut(df["pred_proba"], bins=[0, 0.3, 0.6, 1], labels=["Bajo", "Medio", "Alto"])

# Sidebar de filtros
with st.sidebar:
    st.header("🎚️ Filtros")
    segment_filter = st.selectbox("Segmento", ["Todos"] + list(df["segment"].unique()))
    income_cols = df.filter(like="income_range").columns.tolist()
    income_filter = st.selectbox("Ingreso", ["Todos"] + income_cols)
    risk_cols = df.filter(like="risk_profile").columns.tolist()
    risk_filter = st.selectbox("Perfil de riesgo", ["Todos"] + risk_cols)

# Aplicar filtros
df_filtered = df.copy()
if segment_filter != "Todos":
    df_filtered = df_filtered[df_filtered["segment"] == segment_filter]
if income_filter != "Todos":
    df_filtered = df_filtered[df_filtered[income_filter] == 1]
if risk_filter != "Todos":
    df_filtered = df_filtered[df_filtered[risk_filter] == 1]

# Mostrar métricas clave
st.subheader("📊 Métricas Clave")
col1, col2, col3 = st.columns(3)
col1.metric("Total Clientes", len(df_filtered))
col2.metric("Prob. Promedio", f"{df_filtered['pred_proba'].mean():.2%}")
col3.metric("% Alta Propensión", f"{(df_filtered['segment'] == 'Alto').mean():.2%}")

# Tablas y gráficos en pestañas
tabs = st.tabs(["Clientes", "Distribución", "Importancia (SHAP)"])

with tabs[0]:
    st.subheader("📋 Top Clientes")
    st.dataframe(
        df_filtered[["pred_proba", "segment"] + income_cols + risk_cols]
        .sort_values("pred_proba", ascending=False)
        .head(15)
        .style.format({"pred_proba": "{:.2%}"})
    )

with tabs[1]:
    st.subheader("📈 Distribución de Probabilidades")
    fig, ax = plt.subplots()
    ax.hist(df_filtered["pred_proba"], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Probabilidad de Contratación")
    ax.set_ylabel("Cantidad de Clientes")
    st.pyplot(fig)

with tabs[2]:
    st.subheader("🔍 Explicación del Modelo (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    cliente_idx = st.number_input("Índice del Cliente (0-N)", min_value=0, max_value=len(X)-1, value=0, step=1)
    
    st.write(f"Análisis SHAP para el cliente: {cliente_idx}")
    shap.force_plot(explainer.expected_value, shap_values[cliente_idx,:], X.iloc[cliente_idx,:], matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)

# Footer
st.caption("🚧 Prototipo creado por Pablo Flores | Versión mejorada con SHAP y filtros avanzados")