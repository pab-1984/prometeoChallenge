.PHONY: all eda fe model insights report clean

all: eda fe model insights report

eda:
	@echo "🚀 Ejecutando EDA..."
	python3 scripts/01_eda.py

fe:
	@echo "⚙️  Ejecutando Feature Engineering..."
	python3 scripts/02_feature_engineering.py

model:
	@echo "📊 Ejecutando Modelado..."
	python3 scripts/03_modeling.py

insights:
	@echo "🔍 Ejecutando Business Insights..."
	python3 scripts/04_business_insights.py

report:
	@echo "📝 Generando Reporte Final..."
	python3 reports/report_generator.py

clean:
	@echo "🧹 Limpiando archivos temporales..."
	rm -f reports/final_report.pdf reports/final_report.md
