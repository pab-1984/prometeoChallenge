import subprocess
import os

def run_script(path):
    print(f"\n🚀 Ejecutando {path}")
    result = subprocess.run(["python3", path])
    if result.returncode != 0:
        raise RuntimeError(f"❌ Error en {path}")

def main():
    try:
        # Paso 1: Exploración
        run_script("scripts/01_eda.py")

        # Paso 2: Feature Engineering
        run_script("scripts/02_feature_engineering.py")

        # Paso 3: Modelado
        run_script("scripts/03_modeling.py")

        # Paso 4: Business Insights
        run_script("scripts/04_business_insights.py")

        # Paso 5: Generar Reporte Final
        run_script("reports/report_generator.py")

        print("\n✅ Flujo completo ejecutado exitosamente.")

    except Exception as e:
        print(f"\n❗ ERROR EN EL FLUJO: {e}")

if __name__ == "__main__":
    main()
