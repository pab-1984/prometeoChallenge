# scripts/04_business_insights.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

def run_business_insights():
    # Cargar dataset final para insights
    df = pd.read_csv("data/processed/final_dataset.csv")

    # Separar features y target
    X = df.drop(columns=['has_insurance'])
    y = df['has_insurance']

    # Cargar modelo ajustado
    model = joblib.load("outputs/models/xgboost_tuned_model.pkl")

    # Predecir probabilidades
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    # Crear segmentos de propensión
    bins = [0, 0.3, 0.6, 1]
    labels = ['Baja', 'Media', 'Alta']
    df['propensity_segment'] = pd.cut(df['propensity_score'], bins=bins, labels=labels, include_lowest=True)

    os.makedirs("reports/insights", exist_ok=True)

    # Visualización 1: Distribución de segmentos
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='propensity_segment', palette='Blues')
    plt.title('Distribución de Clientes por Segmento de Propensión')
    plt.xlabel('Segmento de Propensión')
    plt.ylabel('Número de Clientes')
    plt.savefig("reports/insights/distribucion_segmentos.png")
    plt.close()

    # Visualización 2: Edad promedio por segmento
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='propensity_segment', y='age', palette='Blues')
    plt.title('Edad Promedio por Segmento de Propensión')
    plt.xlabel('Segmento de Propensión')
    plt.ylabel('Edad Promedio (estandarizada)')
    plt.savefig("reports/insights/edad_promedio_segmentos.png")
    plt.close()

    # Visualización 3: Gasto promedio mensual por segmento
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='propensity_segment', y='avg_spent', palette='Blues')
    plt.title('Gasto Promedio Mensual por Segmento de Propensión')
    plt.xlabel('Segmento de Propensión')
    plt.ylabel('Gasto Promedio Mensual (estandarizado)')
    plt.savefig("reports/insights/gasto_promedio_segmentos.png")
    plt.close()

    # Visualización 4: Número promedio de productos contratados por segmento
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='propensity_segment', y='product_count', palette='Blues')
    plt.title('Número Promedio de Productos Contratados por Segmento')
    plt.xlabel('Segmento de Propensión')
    plt.ylabel('Número Promedio de Productos (estandarizado)')
    plt.savefig("reports/insights/productos_promedio_segmentos.png")
    plt.close()

    # Análisis SHAP (para entender qué variables impulsan la predicción)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # SHAP summary plot
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig("reports/insights/shap_summary.png")
    plt.close()

    print("✅ Insights de negocio generados y guardados en 'reports/insights/'")

if __name__ == "__main__":
    run_business_insights()
