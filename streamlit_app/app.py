import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


# from src.diagnostics import (
#     render_national_trends,
#     render_anomaly_diagnostics
# )


from src.eda_utils import (
    compute_commodity_dominance,
    get_national_monthly_allocation,
    plot_enhanced_national_trends,
    generate_anomaly_reporting_table,
    plot_state_commodity_trends
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

tab1, tab2 = st.tabs(
    ["Data Visualisation Mini-Dashboard", "ML Forecasting & Visualisation"]
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
# VISUALISATIONS PAGE
# -------------------------------------------------------------------

with tab1:
    
    # DATE DETAILS DISPLAY
    st.header("Dataset Overview")
    earliest_date = df_diag["date"].min()
    latest_date = df_diag["date"].max()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Earliest Date Available", earliest_date.strftime("%b %Y"))

    with col2:
        st.metric("Latest Date Available", latest_date.strftime("%b %Y"))

    # TOTAL NUMBER OF STATE DETAILS DISPLAY
    st.subheader("State Coverage")
    unique_states = sorted(df_diag["state_name"].unique())
    display_states = [state.title() for state in unique_states]
    
    num_states = len(display_states)
    st.metric("Total States / Union Territories Available", num_states)

    with st.expander("View State List"):
        st.write(unique_states)
    
    st.header("National Allocation Summary")

    # -----------------------------------------------
    # Aggregate National Monthly Allocation
    # -----------------------------------------------

    national_monthly = (
        df_diag.groupby(["date", "commodity"], as_index=False)
        .agg(total_allocated_qty=("total_allocated_qty", "sum"))
    )

    pivot_national = (
        national_monthly
        .pivot(index="date", columns="commodity", values="total_allocated_qty")
        .reindex(columns=["rice", "wheat"])
        .fillna(0)
        .reset_index()
        .rename(columns={
            "rice": "rice_allocated",
            "wheat": "wheat_allocated"
        })
    )

    pivot_national["total_allocated_qty"] = (
        pivot_national["rice_allocated"] +
        pivot_national["wheat_allocated"]
    )

    # -----------------------------------------------
    # Averages
    # -----------------------------------------------

    avg_total_nat = pivot_national["total_allocated_qty"].mean()
    avg_rice_nat = pivot_national["rice_allocated"].mean()
    avg_wheat_nat = pivot_national["wheat_allocated"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Total Allocation (National)", f"{avg_total_nat:,.0f}")

    with col2:
        st.metric("Avg Rice Allocation (National)", f"{avg_rice_nat:,.0f}")

    with col3:
        st.metric("Avg Wheat Allocation (National)", f"{avg_wheat_nat:,.0f}")

    # -----------------------------------------------
    # Commodity Dominance
    # -----------------------------------------------

    st.subheader("National Commodity Dominance")

    nat_total = pivot_national["total_allocated_qty"].sum()
    nat_rice = pivot_national["rice_allocated"].sum()
    nat_wheat = pivot_national["wheat_allocated"].sum()

    rice_share_nat = nat_rice / nat_total
    wheat_share_nat = nat_wheat / nat_total

    if rice_share_nat > 0.5:
        dominance_label = "Rice Dominant Region"
    else:
        dominance_label = "Wheat Dominant Region"

    st.markdown(f"### {dominance_label}")
    st.write(f"**Rice Share:** {rice_share_nat*100:.1f}%")
    st.write(f"**Wheat Share:** {wheat_share_nat*100:.1f}%")

    # -----------------------------------------------
    # NATIONAL TREND DISPLAY
    # -----------------------------------------------

    st.header("National Allocation Trends")
    pivot_df = get_national_monthly_allocation(df_diag)
    fig, anomaly_dates = plot_enhanced_national_trends(pivot_df)
    st.pyplot(fig)

    # -----------------------------------------------
    # ANOMALY WINDOWS DISPLAY
    # -----------------------------------------------

    st.subheader("Reporting Diagnostic Around Anomaly Windows")

    anomaly_table = generate_anomaly_reporting_table(
        df_diag,
        anomaly_dates
    )
    
    st.dataframe(anomaly_table)

    # -----------------------------------------------
    # STATE-LEVEL TREND DISPLAY
    # -----------------------------------------------

    # ALLOCATION SUMMARY - MEAN VALUES
    st.subheader("State-Level Allocation Summary")

    state_options = display_states
    selected_state_summary = st.selectbox(
    "Select State for Summary Statistics",
    state_options
    )

    # convert selection back to lowercase for filtering
    df_summary = df_diag[df_diag["state_name"] == selected_state_summary.lower()]
    
    # Aggregate monthly national level
    monthly_summary = (
        df_summary.groupby(["date", "commodity"], as_index=False)
        .agg(total_allocated_qty=("total_allocated_qty", "sum"))
    )
    
    pivot_summary = (
    monthly_summary
    .pivot(index="date", columns="commodity", values="total_allocated_qty")
    .reset_index()
    )
    
    # Ensure both columns exist
    if "rice" not in pivot_summary.columns:
        pivot_summary["rice"] = 0

    if "wheat" not in pivot_summary.columns:
        pivot_summary["wheat"] = 0

    pivot_summary = pivot_summary.rename(
        columns={"rice": "rice_allocated", "wheat": "wheat_allocated"}
    )
    
    pivot_summary["total_allocated_qty"] = (
        pivot_summary["rice_allocated"].fillna(0)
        + pivot_summary["wheat_allocated"].fillna(0)
    )
    
    # Compute averages
    avg_total = pivot_summary["total_allocated_qty"].mean()
    avg_rice = pivot_summary["rice_allocated"].mean()
    avg_wheat = pivot_summary["wheat_allocated"].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Total Allocation (in MT)", f"{avg_total:,.0f}")
    
    with col2:
        st.metric("Average Rice Allocation (in MT)", f"{avg_rice:,.0f}")

    with col3:
        st.metric("Average Wheat Allocation (in MT)", f"{avg_wheat:,.0f}")
    
    # -------------------------------------------------
    # Commodity Dominance Card
    # -------------------------------------------------

    st.subheader("Commodity Dominance")
    dominance = compute_commodity_dominance(pivot_summary)

    st.markdown(
    f"### {dominance['label']}"
    )

    st.write(
    f"**Rice Share:** {dominance['rice_share']*100:.1f}%"
    )

    st.write(
    f"**Wheat Share:** {dominance['wheat_share']*100:.1f}%"
    )

    st.subheader("Rice & Wheat Allocation Trends (State-Level)")

    state_fig = plot_state_commodity_trends(
        df_diag,
        selected_state_summary
    )

    st.pyplot(state_fig)


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


with tab2:

    st.header("📈 State-Level 3-Month Ahead Allocation Forecast")

    # -------------------------------------------------
    # Load forecasting dataset (includes target)
    # -------------------------------------------------
    df_forecast = load_forecasting_data()

    # Ensure delta_target exists (derived feature)
    if "delta_target" not in df_forecast.columns:
        df_forecast["delta_target"] = (
            df_forecast["target"] - df_forecast["total_allocated_qty"]
        )

    # -------------------------------------------------
    # STATE + COMMODITY SELECTION
    # -------------------------------------------------
    available_states = sorted(df_forecast["state"].unique())
    available_commodities = sorted(df_forecast["commodity"].unique())
    #available_commodities.append("both")

    selected_state = st.selectbox("Select State", available_states)
    selected_commodity = st.selectbox("Select Commodity", available_commodities)

    # -------------------------------------------------
    # CUTOFF MONTH SELECTION
    # -------------------------------------------------
    available_dates = (
        df_forecast["date"]
        .sort_values()
        .unique()
    )

    selected_cutoff = st.selectbox(
        "Select Forecast Cutoff (Month)",
        available_dates,
        index=len(available_dates) - 1
    )

    selected_year = selected_cutoff.year
    selected_month = selected_cutoff.month

    st.markdown(
        f"""
        **Selected Configuration:**  
        • State: **{selected_state}**  
        • Commodity: **{selected_commodity}**  
        • Cutoff: **{selected_year}-{selected_month:02d}**
        """
    )

    # -------------------------------------------------
    # RUN FORECAST
    # -------------------------------------------------
    
    if st.button("Run Forecast"):
        
        result = run_forecast_for_cutoff(
            df_forecast,
            selected_year,
            selected_month,
            selected_state,
            selected_commodity,
            numeric_features=[
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_3",
            "rolling_std_3"
        ],
            categorical_features=[]
        )

        if result["status"] == "no_data":
            st.error(result["message"])
            st.stop()

        if result["status"] == "insufficient_data":
            st.warning(result["message"])
            st.stop()

        if result["status"] == "success":
            st.success("Forecast completed successfully.")

            st.subheader("Model Performance Metrics")

            metrics = result["metrics"]
            st.write(metrics)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("MAE", f"{metrics['MAE']:.2f}")

            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")

            with col3:
                st.metric("MAPE (%)",f"{metrics['MAPE']:.2f}")

            with col4:
                st.metric("R2 Score ", f"{metrics['R2']:.2f}")


            val_df = result["val_df"]
            predictions = result["predictions"]

            st.subheader("Validation Data Preview")
            val_preview = result["val_df"].drop(
                 columns=["is_anomaly", "is_valid_data"],
                 errors="ignore"
            )
            st.dataframe(val_preview.head(5))

            st.pyplot(
                plot_actual_vs_predicted(
                    result["train_df"],
                    result["val_df"],
                    result["predictions"],
                    selected_commodity
                )
            )


            if result.get("imputed", False):
                st.info(
                    f"Forward-fill imputation applied for lag month: "
                    f"{result['imputed_month'].date()}"
                )

