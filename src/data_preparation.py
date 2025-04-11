import pandas as pd

def load_datasets(demo_path, products_path, trans_path):
    demographics = pd.read_csv(demo_path)
    products = pd.read_csv(products_path)
    transactions = pd.read_csv(trans_path)
    
    # Formateo de fechas
    transactions["date"] = pd.to_datetime(transactions["date"])
    products["contract_date"] = pd.to_datetime(products["contract_date"])
    
    return demographics, products, transactions


def preprocess_transactions(transactions):
    # Agregar año y mes para análisis temporal
    transactions["year_month"] = transactions["date"].dt.to_period("M")
    return transactions


def generate_product_flags(products):
    # Crear columnas binarias por tipo de producto
    product_flags = products.pivot_table(index="user_id", 
                                         columns="product_type", 
                                         aggfunc="size", 
                                         fill_value=0)
    product_flags.columns = [f"has_{col}" for col in product_flags.columns]
    return product_flags.reset_index()


def merge_datasets(demographics, transactions, products):
    product_flags = generate_product_flags(products)

    # Agregación de transacciones por usuario
    txn_summary = transactions.groupby("user_id").agg(
        total_spent=("amount", "sum"),
        avg_spent=("amount", "mean"),
        txn_count=("amount", "count"),
        favorite_category=("merchant_category", lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    ).reset_index()

    # Merge final
    df = demographics.merge(txn_summary, on="user_id", how="left") \
                     .merge(product_flags, on="user_id", how="left")

    df.fillna({"total_spent": 0, "avg_spent": 0, "txn_count": 0, "favorite_category": "unknown"}, inplace=True)

    return df
