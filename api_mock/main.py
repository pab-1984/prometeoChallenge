from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

app = FastAPI(title="Mocked Open Banking API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"


demographics_df = pd.read_csv(DATA_DIR / "demographics.csv")
products_df = pd.read_csv(DATA_DIR / "products.csv")
transactions_df = pd.read_csv(DATA_DIR / "transactions.csv")


@app.get("/")
def root():
    return {"message": "Mock API para extracción de datos"}

@app.get("/demographics/")
def get_demographics():
    return demographics_df.to_dict(orient="records")

@app.get("/products/")
def get_products():
    return products_df.to_dict(orient="records")

@app.get("/transactions/")
def get_transactions():
    return transactions_df.to_dict(orient="records")
