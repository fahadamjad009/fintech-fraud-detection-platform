\# System Architecture — FinTech Fraud Detection Platform



This project is structured like a production ML workflow: \*\*offline training\*\*, \*\*decisioning logic\*\*, \*\*monitoring\*\*, and \*\*serving\*\*.



\## High-level architecture



```mermaid

flowchart LR

&nbsp;   A\[Raw Dataset<br/>creditcard.csv] --> B\[Ingestion<br/>src/ingestion]

&nbsp;   B --> C\[Processing + Features<br/>src/features]

&nbsp;   C --> D\[Processed Dataset<br/>data/processed/\*.parquet]



&nbsp;   D --> E\[Training (Baseline LogReg)<br/>src/models/train\_baseline.py]

&nbsp;   D --> F\[Training (XGBoost)<br/>src/models/train\_xgb.py]



&nbsp;   E --> M\[Saved Model Pipeline<br/>models/baseline\_logreg.joblib]

&nbsp;   F --> N\[Saved Model<br/>models/xgb.joblib]



&nbsp;   E --> R\[Reports + Metrics<br/>reports/\*.json]

&nbsp;   F --> R

&nbsp;   E --> P\[Figures<br/>reports/figures/\*.png]

&nbsp;   F --> P



&nbsp;   R --> G\[Cost Optimisation<br/>src/models/cost\_threshold\_optimiser.py]

&nbsp;   G --> RC\[Cost Outputs<br/>reports/cost\_threshold\_optimum.json<br/>reports/figures/cost\_curve\_by\_threshold.png]



&nbsp;   R --> S\[Streamlit Dashboard<br/>dashboards/app.py]

&nbsp;   P --> S

&nbsp;   RC --> S



&nbsp;   M --> API\[FastAPI Scoring<br/>api/app.py]

&nbsp;   API --> OUT\[Prediction Output<br/>proba + label + threshold]



&nbsp;   D --> MON\[Monitoring / Drift<br/>src/monitoring]

&nbsp;   MON --> MR\[Monitoring Reports<br/>reports/monitoring/\*.json]

&nbsp;   MR --> S



