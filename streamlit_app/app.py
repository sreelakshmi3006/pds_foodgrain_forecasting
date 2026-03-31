import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.eda_utils import (
    commodity_dominance_calculator,
    plot_enhanced_national_trends,
    national_anomaly_reporting_table,
    plot_yearly_national_allocation,
    plot_yearly_state_allocation,
    plot_enhanced_state_trends,
    state_anomaly_reporting_table
)

from src.geo_utils import get_state_codes

from src.forecasting import run_forecast_for_cutoff
from src.plotting import plot_state_prediction


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
# LOAD CLEANED DATA
# -------------------------------------------------------------------

@st.cache_data
def load_visualisation_data():

    project_root = Path(__file__).resolve().parents[1]

    national_path = project_root / "data" / "cleaned" / "National_Aggregate_Allocations_Cleaned.csv"
    state_path = project_root / "data" / "cleaned" / "State_Level_Allocations_Cleaned.csv"

    national_df = pd.read_csv(national_path, parse_dates=["date"])
    state_df = pd.read_csv(state_path, parse_dates=["date"])

    return national_df, state_df


national_df, state_df = load_visualisation_data()


# -------------------------------------------------------------------
# VISUALISATIONS PAGE
# -------------------------------------------------------------------

with tab1:

    st.header("Data Visualisation Mini-Dashboard")

    national_tab, state_tab = st.tabs(
        ["National Analysis", "State Analysis"]
    )

    # ===============================================================
    # NATIONAL ANALYSIS
    # ===============================================================

    with national_tab:

        st.subheader("National Commodity Dominance")

        col1, col2 = st.columns([2,1])

        with col1:
            fig, dominant, share = commodity_dominance_calculator(national_df)
            st.pyplot(fig)

        with col2:
            st.metric(
                label="Dominant Commodity",
                value=dominant,
                delta=f"{share:.1f}% share"
            )

        st.divider()

        # -------------------------------------------------

        st.subheader("Year-wise National Allocation")

        fig = plot_yearly_national_allocation(national_df)
        st.pyplot(fig)

        st.divider()

        # -------------------------------------------------

        st.subheader("Enhanced National Allocation Trends")

        fig, anomaly_dates = plot_enhanced_national_trends(national_df)
        st.pyplot(fig)

        st.divider()

        # -------------------------------------------------

        st.subheader("Anomaly Reporting")

        anomaly_table = national_anomaly_reporting_table(national_df)
        st.dataframe(anomaly_table, use_container_width=True)


    # ===============================================================
    # STATE ANALYSIS
    # ===============================================================

    with state_tab:

        st.header("State Analysis")

        # -------------------------------------------------
        # STATE SELECTOR
        # -------------------------------------------------

        state_dict = get_state_codes(state_df)

        state_display = {
            f"{name} ({code})": code
            for code, name in state_dict.items()
        }

        selected = st.selectbox(
            "Select State",
            list(state_display.keys())
        )

        state_code = state_display[selected]
        state_name = state_dict[state_code]

        state_data = state_df[state_df["state_code"] == state_code]

        st.divider()

        # -------------------------------------------------

        st.subheader(f"{state_name} ({state_code})")

        # -------------------------------------------------
        # DOMINANCE
        # -------------------------------------------------

        col1, col2 = st.columns([2,1])

        with col1:
            fig, dominant, share = commodity_dominance_calculator(state_data)
            st.pyplot(fig)

        with col2:
            st.metric(
                label="Dominant Commodity",
                value=dominant,
                delta=f"{share:.1f}% share"
            )

        st.divider()

        # -------------------------------------------------

        st.subheader("Year-wise Allocation")

        fig = plot_yearly_state_allocation(state_data)
        st.pyplot(fig)

        st.divider()

        # -------------------------------------------------

        st.subheader("Enhanced Allocation Trends")

        fig, anomaly_dates = plot_enhanced_state_trends(state_data)
        st.pyplot(fig)

        st.divider()

        # -------------------------------------------------

        st.subheader("Anomaly Reporting")

        anomaly_table = state_anomaly_reporting_table(state_data)
        st.dataframe(anomaly_table, use_container_width=True)



@st.cache_data
def load_forecasting_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        project_root
        / "data"
        / "preprocessed"
        / "state_level_features.csv"
    )

    df = pd.read_csv(data_path, parse_dates=["date"])

    return df.sort_values(
        ["state_code", "commodity", "date"]
    ).reset_index(drop=True)

df_forecast = load_forecasting_data()

# -------------------------------------------------------------------
# FORECASTING TAB
# -------------------------------------------------------------------

with tab2:

    st.header(" State-Level 3-Month Ahead Allocation Forecast")
    
    # Feature Columns
    
    feature_cols = [
        'lag_1', 'lag_2', 'lag_3',
        'lag_6', 'lag_9', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6',
        'rolling_std_3'
    ]

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------
    # Use existing geo_utils function
    state_dict = get_state_codes(df_forecast)
    state_display = {
        f"{name} ({code})": code
        for code, name in state_dict.items()
    }
    
    selected_display = st.selectbox("Select State", list(state_display.keys()))

    selected_state = state_display[selected_display]
    state_name = state_dict[selected_state]


    available_commodities = sorted(
        df_forecast[df_forecast["state_code"] == selected_state]["commodity"].unique()
    )

    commodity_display = [c.capitalize() for c in available_commodities]
    selected_commodity_display = st.selectbox("Select Commodity", commodity_display)

    selected_commodity = selected_commodity_display.lower()
    

    df_filtered = df_forecast[
        (df_forecast["state_code"] == selected_state) &
        (df_forecast["commodity"] == selected_commodity)
    ]

    available_dates = sorted(df_filtered["date"].unique())

    # Find earliest date for this state-commodity
    min_date = df_filtered["date"].min()

    # Valid cutoff must be >= 12 months after start
    min_valid_date = min_date + pd.DateOffset(months=12)

    # Filter valid dates
    valid_dates = [d for d in available_dates if d >= min_valid_date]

    # Safety check (edge case)
    if not valid_dates:
        st.error("Not enough data to perform forecasting (requires at least 12 months history).")
        st.stop()

    # Create display mapping (Month Year)
    date_display = {
        d.strftime("%b %Y"): d
        for d in valid_dates
    }

    selected_date_display = st.selectbox(
        "Select Forecast Cutoff (Month)",
        list(date_display.keys())
    )

    selected_date = date_display[selected_date_display]

    selected_year = pd.to_datetime(selected_date).year
    selected_month = pd.to_datetime(selected_date).month

    # --------------------------------------------------
    # Show Selection
    # --------------------------------------------------
    st.markdown("### Selected Configuration:")

    cutoff_str = selected_date.strftime("%b %Y")
    t_plus_1 = (selected_date + pd.DateOffset(months=1)).strftime("%b %Y")
    t_plus_2 = (selected_date + pd.DateOffset(months=2)).strftime("%b %Y")

    st.write(f"• State: {state_name} ({selected_state})")
    st.write(f"• Commodity: {selected_commodity_display}")
    st.write(f"• Predicting Months: {cutoff_str}, {t_plus_1}, {t_plus_2}")

    # --------------------------------------------------
    # Run Forecast
    # --------------------------------------------------
    if st.button("Run Forecast"):

        result = run_forecast_for_cutoff(
            df_forecast,
            selected_year,
            selected_month,
            selected_state,
            selected_commodity,
            feature_cols,
            horizon=3
        )

        # ------------------------------------------
        # Status Handling
        # ------------------------------------------
        if result["status"] == "ERROR":
            st.error(result["message"])
            st.stop()

        elif result["status"] == "WARNING":
            st.warning(result["message"])

        elif result["status"] == "SUCCESS":
            st.success("Forecast completed successfully.")

        # ------------------------------------------
        # Prepare Data
        # ------------------------------------------
        preds = result["predictions"]

        selected_date = pd.Timestamp(
            year=selected_year,
            month=selected_month,
            day=1
        ) + pd.offsets.MonthEnd(0)

        df_sc = df_forecast[
            (df_forecast["state_code"] == selected_state) &
            (df_forecast["commodity"] == selected_commodity)
        ]

        forecast_dates = [
            selected_date + pd.DateOffset(months=i)
            for i in range(len(preds))
        ]

        # ------------------------------------------
        # Forecast Table
        # ------------------------------------------
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted Allocation": preds
        })

        actuals = []
        for d in forecast_dates:
            row = df_sc[df_sc["date"] == d]
            if not row.empty:
                actuals.append(row["total_allocated_qty"].values[0])
            else:
                actuals.append(None)

        forecast_df["Actual Allocation"] = actuals

        st.subheader("3-Month Forecast")
        st.dataframe(
            forecast_df.style.format({
                "Predicted Allocation": "{:,.0f}",
                "Actual Allocation": "{:,.0f}"
            })
        )

        # ------------------------------------------
        # Metrics (only meaningful for first step)
        # ------------------------------------------
        metrics = result.get("metrics")

        if metrics:
            st.subheader("Model Performance")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("MAPE (%)", f"{metrics['MAPE'] * 100:.2f}")

            with col2:
                st.metric("R2 Score", f"{metrics['R2']:.2f}")

        # ------------------------------------------
        # Plot
        # ------------------------------------------
        st.subheader("Allocation Trend with Forecast")

        fig = plot_state_prediction(
            df_sc,
            forecast_dates,
            preds,
            selected_state,
            selected_commodity
        )

        st.pyplot(fig)