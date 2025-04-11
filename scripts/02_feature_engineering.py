# scripts/02_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def run_feature_engineering():
    
    df = pd.read_csv("data/processed/clientes_unificados.csv")
    transactions_df = pd.read_csv("data/raw/transactions.csv")
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])

   
    if "has_insurance" not in df.columns:
        df["has_insurance"] = 0
    df["has_insurance"] = df["has_insurance"].astype(int)

    
    df["product_count"] = df.filter(like="has_").drop(columns=["has_insurance"]).sum(axis=1)

    
    top_cats = df["favorite_category"].value_counts().index[:5]
    df["favorite_category_enc"] = df["favorite_category"].apply(lambda x: x if x in top_cats else "other")


    reference_date = transactions_df["date"].max() + pd.Timedelta(days=1)
    rfm_df = transactions_df.groupby('user_id').agg(
        recency=('date', lambda x: (reference_date - x.max()).days),
        frequency=('transaction_id', 'count'),
        monetary=('amount', 'sum')
    ).reset_index()

    df = pd.merge(df, rfm_df, on='user_id', how='left')
    df['recency'] = df['recency'].fillna(999)
    df['frequency'] = df['frequency'].fillna(0)
    df['monetary'] = df['monetary'].fillna(0)


    transactions_df['month'] = transactions_df['date'].dt.to_period('M')
    monthly_spending = transactions_df.groupby(['user_id', 'month'])['amount'].sum().reset_index()

    spending_volatility = monthly_spending.groupby('user_id')['amount'].std(ddof=0).reset_index()
    spending_volatility.rename(columns={'amount': 'spending_volatility'}, inplace=True)

    df = pd.merge(df, spending_volatility, on='user_id', how='left')
    df['spending_volatility'] = df['spending_volatility'].fillna(0)


    occupation_counts = df['occupation'].value_counts()
    threshold = 5
    rare_occupations = occupation_counts[occupation_counts < threshold].index
    df['occupation_enc'] = df['occupation'].replace(rare_occupations, 'Other_Occupation')


    df['age_x_product_count'] = df['age'] * df['product_count']
    df['avg_spent_x_frequency'] = df['avg_spent'] * df['frequency']


    categorical_vars = ["income_range", "risk_profile", "favorite_category_enc", "occupation_enc"]
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)


    drop_cols = ["user_id", "favorite_category", "occupation"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)


    num_cols = [
        "age", "total_spent", "avg_spent", "txn_count", "product_count",
        "recency", "frequency", "monetary", "spending_volatility",
        "age_x_product_count", "avg_spent_x_frequency"
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])


    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("✅ Dataset para modelado guardado en data/processed/final_dataset.csv")

if __name__ == "__main__":
    run_feature_engineering()
