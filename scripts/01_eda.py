import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_preparation import load_datasets, preprocess_transactions, merge_datasets

def run_eda():
    
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

    
    demo_path = "data/raw/demographics.csv"
    products_path = "data/raw/products.csv"
    trans_path = "data/raw/transactions.csv"

    demographics, products, transactions = load_datasets(demo_path, products_path, trans_path)
    transactions = preprocess_transactions(transactions)
    df_model = merge_datasets(demographics, transactions, products)
    
    print("\n Análisis de Ocupación:")
    
    top_occupations = df_model['occupation'].value_counts().head(15)
    print("Top 15 Ocupaciones:\n", top_occupations)

    
    plt.figure(figsize=(12, 8))
    avg_spent_by_occupation = df_model[df_model['occupation'].isin(top_occupations.index)] \
        .groupby('occupation')['avg_spent'].mean().sort_values(ascending=False)
    sns.barplot(y=avg_spent_by_occupation.index, x=avg_spent_by_occupation.values)
    plt.title('Gasto Promedio Mensual por Ocupación (Top 15)')
    plt.xlabel('Gasto Promedio Mensual')
    plt.ylabel('Ocupación')
    plt.tight_layout()
    plt.savefig("outputs/figures/gasto_vs_ocupacion_top15.png")
    plt.close()
    print("Gráfico Gasto vs Ocupación guardado.")

    # ----------------------------
    print("\n Análisis de Combinaciones de Productos:")
    product_cols = [col for col in df_model.columns if col.startswith('has_') and col != 'has_insurance']
    df_model['product_count_eda'] = df_model[product_cols].sum(axis=1)
    print("Distribución de cantidad de productos por cliente:\n", df_model['product_count_eda'].value_counts())

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_model, x='product_count_eda')
    plt.title('Distribución de Cantidad de Productos por Cliente')
    plt.xlabel('Número de Productos Contratados')
    plt.ylabel('Número de Clientes')
    plt.savefig("outputs/figures/distribucion_num_productos.png")
    plt.close()
    print("Gráfico Distribución Número de Productos guardado.")

    # Combinaciones específicas (si pocas)
    if len(product_cols) < 5:
        product_combinations = df_model[product_cols].astype(str).agg('-'.join, axis=1)
        common_combinations = product_combinations.value_counts().head(10)
        print("\nCombinaciones de productos más comunes:\n", common_combinations)
    
    os.makedirs("data/processed", exist_ok=True)
    df_model.to_csv("data/processed/clientes_unificados.csv", index=False)
    print("✅ Dataset unificado guardado.")

    
    print(df_model.head())
    print(df_model.describe(include="all"))

    
    df_model[["age", "total_spent", "avg_spent", "txn_count"]].hist(bins=20)
    plt.suptitle("Distribuciones numéricas")
    plt.tight_layout()
    plt.savefig("outputs/figures/histogramas_numericos.png")

    
    order = ["<30k", "30k-50k", "50k-100k", "100k-150k", ">150k"]
    plt.figure()
    sns.barplot(data=df_model, x="income_range", y="avg_spent", ci=None, order=order)
    plt.title("Gasto promedio mensual vs. ingreso declarado")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/gasto_vs_ingreso.png")

    
    corr = df_model[["age", "total_spent", "avg_spent", "txn_count"]].corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlación entre variables numéricas")
    plt.tight_layout()
    plt.savefig("outputs/figures/heatmap_correlaciones.png")

    print("📊 Visualizaciones guardadas en outputs/figures/")

if __name__ == "__main__":
    run_eda()
