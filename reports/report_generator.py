import pandas as pd
import joblib
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os

def draw_paragraph(c, text, x, y, max_width=500, line_height=14):
    from textwrap import wrap
    for line in wrap(text, width=100):
        c.drawString(x, y, line)
        y -= line_height
    return y

def generate_pdf_report(data_path, model_path, output_path="reports/final_report.pdf"):
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    df["pred_proba"] = model.predict_proba(df.drop(columns=["has_insurance"]))[:, 1]

    top10 = df.sort_values("pred_proba", ascending=False).head(10)
    avg_proba = df["pred_proba"].mean()
    high_propensity = (df["pred_proba"] > 0.6).sum()

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    y = height - 40

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "📊 Reporte de Cross-Selling de Seguros")
    y -= 30

    c.setFont("Helvetica", 12)
    y = draw_paragraph(c, f"Fecha: {datetime.date.today()}", 40, y)
    y = draw_paragraph(c, f"Promedio de propensión: {avg_proba:.2f}", 40, y)
    y = draw_paragraph(c, f"Clientes con alta propensión (> 60%): {high_propensity}", 40, y)
    y -= 20

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Top 10 Clientes Priorizados")
    y -= 20

    c.setFont("Helvetica", 10)
    for idx, row in top10.iterrows():
        y = draw_paragraph(c, f"Cliente #{idx} - Probabilidad: {row['pred_proba']:.2f}", 50, y)
        if y < 100:
            c.showPage()
            y = height - 40

    # Recomendaciones
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Recomendaciones")
    y -= 20
    c.setFont("Helvetica", 12)
    recomendaciones = [
        "📌 Focalizar campañas en clientes con probabilidad > 60%.",
        "📌 Personalizar ofertas por categoría favorita e ingresos.",
        "📌 Considerar seguros como upgrade de productos actuales.",
    ]
    for rec in recomendaciones:
        y = draw_paragraph(c, rec, 50, y)
        y -= 5

    # Agregar imágenes desde folders
    def add_images_from_folder(folder_path, title):
        nonlocal y
        files = sorted([f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg"))])
        if files:
            y -= 20
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, title)
            y -= 20
            for fimg in files:
                img_path = os.path.join(folder_path, fimg)
                try:
                    img = ImageReader(img_path)
                    iw, ih = img.getSize()
                    ratio = min(400 / iw, 300 / ih)
                    iw *= ratio
                    ih *= ratio
                    if y - ih < 60:
                        c.showPage()
                        y = height - 40
                    c.drawImage(img_path, 40, y - ih, width=iw, height=ih)
                    y -= ih + 20
                except Exception as e:
                    y = draw_paragraph(c, f"[Error al cargar imagen: {fimg}]", 50, y)
                    y -= 10

    add_images_from_folder("outputs/figures", "Gráficos Generales")
    add_images_from_folder("reports/insights", "Gráficos de Insight")

    c.save()
    print(f"✅ PDF generado exitosamente en: {output_path}")

# Ejecutar
if __name__ == "__main__":
    generate_pdf_report(
        data_path="data/processed/final_dataset.csv",
        model_path="outputs/models/xgboost_tuned_model.pkl"
    )
