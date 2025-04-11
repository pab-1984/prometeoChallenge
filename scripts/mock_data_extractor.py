import requests
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000"
RAW_DATA_DIR = Path("data/raw/")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

endpoints = {
    "demographics": "/demographics/",
    "products": "/products/",
    "transactions": "/transactions/"
}

def fetch_and_save(endpoint_name, endpoint_path):
    print(f"🔄 Extrayendo {endpoint_name}...")
    url = f"{API_URL}{endpoint_path}"
    response = requests.get(url)
    response.raise_for_status()  # Lanza excepción si hay error HTTP

    data = response.json()
    df = pd.DataFrame(data)
    output_file = RAW_DATA_DIR / f"{endpoint_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Guardado: {output_file}")

def run_mock_extraction():
    for name, path in endpoints.items():
        fetch_and_save(name, path)
    print("🚀 Extracción completa.")

if __name__ == "__main__":
    run_mock_extraction()
