📊 India PDS Foodgrain Allocation Forecasting

A complete end-to-end data science project analyzing and forecasting Public Distribution System (PDS) foodgrain allocations (Rice & Wheat) across Indian states.

🚀 Project Overview

This project builds a state-level forecasting system for foodgrain allocation using:

Historical allocation data
Time-series feature engineering
Machine learning (Random Forest)
Interactive Streamlit dashboard

🎯 Objectives
Analyze national and state-level allocation trends
Detect anomalies in reporting and distribution
Forecast future allocations (3-month horizon)
Build a production-ready data pipeline and dashboard

🧱 Project Structure
pds_foodgrain_forecasting/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── preprocessed/
│
├── notebooks/
│   ├── 01_primary_data_cleaning.ipynb
│   ├── 02_eda_visualisation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_implementation.ipynb
│   └── 05_parameterised_modeling_engine.ipynb
│
├── src/
│   ├── eda_utils.py
│   ├── preprocessing.py
│   ├── forecasting.py
│   ├── plotting.py
│   ├── geo_utils.py
│   ├── text_cleaning.py
│   └── date_utils.py
│
├── app.py
└── README.md

🔄 Workflow
1. Data Cleaning
Standardized column names and text fields
Mapped state names to codes
Created consistent monthly time index
2. Exploratory Data Analysis (EDA)
Commodity dominance (Rice vs Wheat)
Year-wise allocation trends
State-level allocation patterns
Anomaly detection:
Missing reporting states
Sudden allocation changes
3. Feature Engineering
Lag features: lag_1, lag_2, lag_3, lag_6, lag_9, lag_12
Rolling statistics:
rolling_mean_3, rolling_mean_6
rolling_std_3
Time-series safe preprocessing:
Full timeline creation
Forward/backward fill
Outlier clipping
4. Machine Learning
Model: Random Forest Regressor
State + commodity specific training
Recursive forecasting for 3 months
5. Streamlit Dashboard

📌 Features:
National Analysis
Commodity dominance
Year-wise allocation
Trend + anomaly visualization
State Analysis
State selector
Allocation trends
Anomaly reporting
Forecasting
Select state & commodity
Choose forecast cutoff date
3-month prediction
Model performance metrics (MAPE, R²)
Visualization of predictions vs actuals

🧠 Key Design Decisions
Lag-based features → captures temporal dependencies
No imputation of anomalies → preserves real-world signals
Recursive forecasting → realistic multi-step prediction

📈 Model Features
[
 'lag_1', 'lag_2', 'lag_3',
 'lag_6', 'lag_9', 'lag_12',
 'rolling_mean_3', 'rolling_mean_6',
 'rolling_std_3'
]

⚠️ Constraints & Validations
Minimum 12 months history required (lag_12 constraint)
Forecast only allowed for valid cutoff dates
Warning triggered for high volatility regions

🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib
Streamlit

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run Streamlit app
streamlit run app.py

📊 Example Use Case
Select: Kerala (KL) → Rice
Choose cutoff: Jan 2021
Get:
Feb, Mar, Apr predictions
Model accuracy metrics
Visual trend comparison

📌 Key Learnings
Handling panel time-series data
Preventing data leakage in lag features
Designing modular ML pipelines
Building production-ready dashboards
Managing real-world messy datasets

🔮 Future Improvements
Add seasonality features (month encoding)
Try advanced models (XGBoost, LSTM)
Add district-level granularity (optional)
Improve anomaly detection (statistical methods)

👤 Author

Sreelakshmi S
Aspiring Data Analyst | Data Scientist
Focused on building real-world, production-ready data projects