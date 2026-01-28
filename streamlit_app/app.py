import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.diagnostics import (
    render_national_trends,
    render_anomaly_diagnostics
)


from src.forecasting import run_forecast_for_cutoff
from src.plotting import plot_actual_vs_predicted


# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------

st.set_page_config(
    page_title="India PDS Allocation Dashboard",
    layout="wide"
)

st.title("India PDS Rice & Wheat Allocation Dashboard")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Diagnostics", "Forecasting"]
)

# -------------------------------------------------------------------
# LOAD DIAGNOSTIC  DATA
# -------------------------------------------------------------------

@st.cache_data
def load_diagnostics_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cleaned" / "primary_data_clean.csv"

    return pd.read_csv(data_path,parse_dates=["date"])

# Restrict to reliable analysis window
df_diag = load_diagnostics_data()
df_diag = df_diag[df_diag["date"] >= "2017-10-01"]


# -------------------------------------------------------------------
# DIAGNOSTICS PAGE
# -------------------------------------------------------------------

if page == "Diagnostics":
    st.header("National Allocation Trends")
    st.pyplot(render_national_trends(df_diag))

    st.header("Allocation Anomaly Diagnostic (Jul 2020 - Jan 2021)")
    st.pyplot(render_anomaly_diagnostics(df_diag))

# -------------------------------------------------------------------
# LOAD FORECASTING  DATA
# -------------------------------------------------------------------

@st.cache_data
def load_forecasting_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        project_root
        / "data"
        / "processed"
        / "data_for_model_with_validation_flags.csv"
    )

    return pd.read_csv(data_path, parse_dates=["date"])


if page == "Forecasting":

    st.header("📈 3-Month Ahead Allocation Forecast")

        # ------------------------------------------------------------------
        # Ensure delta_target exists (derived feature)
        # ------------------------------------------------------------------
    
    df_forecast = load_forecasting_data()

    if "delta_target" not in df_forecast.columns:
        df_forecast["delta_target"] = (df_forecast["target"] - df_forecast["total_allocated_qty"])


    # Available months for selection
    available_periods = (
    df_forecast
    .dropna(subset=["delta_target"])
    ["date"]
    .sort_values()
    .dt.to_period("M")
    .astype(str)
    .unique()
    )

    # delta_target is not persisted in CSV by design
    # It is derived at runtime to keep the dataset generic


    selected_period = st.selectbox(
        "Select forecast cutoff (YYYY-MM)",
        available_periods,
        index=len(available_periods) - 1
    )

    year, month = map(int, selected_period.split("-"))

    st.markdown(
        f"""
        **Selected cutoff:**  
        🗓️ {year}-{month:02d}
        """
    )

    if st.button("🚀 Run Forecast"):
        with st.spinner("Running forecasting engine..."):
            results = run_forecast_for_cutoff(
                df_forecast,
                year=year,
                month=month,
                numeric_features=[
                    "lag_1",
                    "lag_2",
                    "lag_3",
                    "rolling_mean_3",
                    "rolling_std_3",
                ],
                categorical_features=["state"],
            )

        st.subheader("🌾 Rice Forecast")

        rice = results["rice"]
        rice_metrics = rice["metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{rice_metrics['R2']:.3f}")
        c2.metric("Normalized MAE", f"{rice_metrics['nMAE']*100:.2f}%")
        c3.metric("Normalized RMSE", f"{rice_metrics['nRMSE']:.3f}")

        fig_rice = plot_actual_vs_predicted(
                rice["train_df"],
                rice["val_df"],
                rice["level_pred"],
                commodity="rice",
            )

        st.pyplot(fig_rice)

        st.subheader("🌾 Wheat Forecast")

        wheat = results["wheat"]
        wheat_metrics = wheat["metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{wheat_metrics['R2']:.3f}")
        c2.metric("Normalized MAE", f"{wheat_metrics['nMAE']*100:.2f}%")
        c3.metric("Normalized RMSE", f"{wheat_metrics['nRMSE']:.3f}")

        fig_wheat = plot_actual_vs_predicted(
                wheat["train_df"],
                wheat["val_df"],
                wheat["level_pred"],
                commodity="wheat",
            )

        st.pyplot(fig_wheat)



