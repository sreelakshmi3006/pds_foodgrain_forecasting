# 📦 PDS Foodgrain (Rice & Wheat) Forecasting — India

This project builds an end-to-end data science and machine learning pipeline to analyse and forecast rice and wheat allocations under India’s Public Distribution System (PDS) using real-world government data.

It combines:
- rigorous data cleaning and validation,
- anomaly-aware feature engineering,
- leakage-safe time-series modeling,
- delta-based forecasting,
- and an interactive Streamlit dashboard.

## 🎯 Problem Context

India’s PDS allocation data presents several real-world challenges:

- reporting delays and administrative shocks (e.g. COVID-era under-reporting),
- incomplete state-level panels,
- strong regional heterogeneity between rice and wheat,
- non-stationary allocation levels.
- Naïve forecasting approaches fail under these conditions.

This project addresses these issues using careful data diagnostics, explicit anomaly handling, and robust ML design choices.

## 🧠 Methodology Overview

### Data

- Monthly, state-level allocation data
- Commodities: Rice and Wheat
- Period: ~2018–2021
- Source: Government of India administrative datasets

### Key Modeling Decisions

- Forecast horizon: 3 months ahead
- Target formulation: Forecasting change in allocation (delta_target) instead of raw levels
- Models: Random Forest Regressor
- Validation: Time-based, leakage-safe splits
- Anomaly handling: Explicit flags; excluded from training
- Panel completeness: Commodity-specific thresholds (rice vs wheat)

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit application with two main sections:

### 1️⃣ Diagnostics

- National-level allocation trends
- Anomaly detection and visualization
- Zoomed views of reporting shock periods

### 2️⃣ Forecasting

- User-selected forecast cutoff (month & year)
- Automatic train–validation split (no hard-coding)
- Separate rice and wheat models
- Normalized evaluation metrics:
- R²
- Normalized MAE
- Normalized RMSE
- Actual vs predicted visualizations

## 🗂️ Project Structure

PDS_FOODGRAIN_FORECASTING/
│
├── data/
│   ├── raw/                # Raw downloaded data (not tracked)
│   ├── cleaned/            # Cleaned intermediate data (not tracked)
│   └── processed/          # Feature-engineered data (not tracked)
│
├── notebooks/
│   ├── 01_primary_data_cleaning.ipynb
│   ├── 02_supplementary_data_cleaning.ipynb
│   ├── 03_eda_visualisation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_ml_modeling.ipynb
│   ├── 06_model_improvement.ipynb
│   ├── 07_alternative_modeling_strategies.ipynb
│   └── 08_parameterised_delta_forecasting_engine.ipynb
│
├── src/
│   ├── diagnostics.py      # National trends & anomaly diagnostics
│   ├── forecasting.py      # ML pipelines, splits, evaluation
│   ├── plotting.py         # Forecast visualisations
│   ├── eda_utils.py        # EDA helper functions
│   ├── date_utils.py       # Date & time utilities
│   ├── geo_utils.py        # State/region helpers
│   └── text_cleaning.py    # Text standardisation utilities
│
├── streamlit_app/
│   └── app.py              # Streamlit application entry point
│
├── powerbi/                # Power BI exploratory dashboards
│
├── requirements.txt
└── README.md

- Note: Data files are intentionally excluded from version control due to size and licensing constraints.
- All preprocessing and feature engineering steps are fully reproducible using the notebooks.

## ⚙️ Environment Setup

python -m venv penv
.\penv\Scripts\Activate.ps1
pip install -r requirements.txt

## ▶️ Run the Streamlit App
- From the project root:
streamlit run streamlit_app\app.py

## 📈 Key Insights

- Forecasting deltas is more stable than forecasting absolute levels
- Leakage prevention materially affects model evaluation
- Commodity-specific modeling is essential for PDS data
- Data diagnostics and validation matter more than complex models

## 🚀 Possible Extensions

- Confidence intervals for forecasts
- Comparison with linear / boosting models
- Automated retraining pipelines
- Public deployment via Streamlit Cloud

## 📜 Disclaimer

This project is for educational and analytical purposes only.
It does not represent official forecasts or policy recommendations.

## 🧑‍💻 Author

Sreelakshmi S
(Data Science & Machine Learning Project)