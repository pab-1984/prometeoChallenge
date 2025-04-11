# Open Banking Challenge

Este proyecto aborda un caso práctico de **Data Science** para una fintech que opera bajo el marco de **Open Banking**. El objetivo es identificar oportunidades de **cross-selling** de productos financieros mediante el análisis de:

- Comportamiento de transacciones bancarias.
- Productos financieros actuales contratados por los usuarios.
- Datos demográficos de los clientes.

---

## Objetivo del Proyecto

Construir un modelo predictivo que permita determinar qué clientes tienen mayor probabilidad de adquirir un **nuevo producto financiero**, como por ejemplo un **seguro**.

---

## Estructura del Proyecto

```bash
open_banking_cross_selling/
├── data/
│   ├── raw/              # Datos originales (CSV)
│   ├── processed/        # Datos limpios y combinados
│   └── external/         # Datos simulados o externos
├── notebooks/            # Jupyter Notebooks ordenados por etapa
├── scripts/              # Scripts de procesamiento de datos
├── src/                  # Scripts reutilizables de procesamiento y modelado
├── outputs/
│   ├── figures/          # Visualizaciones generadas
│   └── models/           # Modelos entrenados
├── reports/              # Informe final del proyecto
├── requirements.txt      # Librerías necesarias
└── README.md             # Este archivo
```