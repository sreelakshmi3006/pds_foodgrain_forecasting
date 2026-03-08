"""
EDA utility functions for PDS Rice & Wheat project.

Reusable aggregation and plotting logic for exploratory analysis.
Framework-agnostic: can be used in notebooks and Streamlit.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates
import matplotlib.ticker as mticker



# -------------------------------------------------------------------
# NATIONAL-LEVEL AGGREGATION
# -------------------------------------------------------------------

def get_national_monthly_allocation(df):
    """
    Aggregates national monthly allocation by commodity.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with columns:
        - date (month start)
        - commodity ('rice', 'wheat')
        - total_allocated_qty

    Returns
    -------
    pd.DataFrame
        Columns:
        - date
        - rice_allocated
        - wheat_allocated
        - total_allocated_qty
    """
    monthly = (
        df.groupby(["date", "commodity"], as_index=False)
          .agg(total_allocated_qty=("total_allocated_qty", "sum"))
    )

    pivot = (
        monthly
        .pivot(index="date", columns="commodity", values="total_allocated_qty")
        .reset_index()
        .rename(columns={"rice": "rice_allocated", "wheat": "wheat_allocated"})
    )

    pivot["total_allocated_qty"] = (
        pivot["rice_allocated"].fillna(0) +
        pivot["wheat_allocated"].fillna(0)
    )

    return pivot.sort_values("date")

# -------------------------------------------------
# DOMINANCE CLASSIFICATION CALACULATION
# -------------------------------------------------

def compute_commodity_dominance(pivot_summary):
    """
    Determines whether a region is Rice or Wheat dominant
    based on total allocation share.
    """
    total_sum = pivot_summary["total_allocated_qty"].sum()
    rice_sum = pivot_summary["rice_allocated"].sum()
    wheat_sum = pivot_summary["wheat_allocated"].sum()

    if total_sum == 0:
        return {
            "label": "No Allocation Data",
            "rice_share": 0,
            "wheat_share": 0
        }

    rice_share = rice_sum / total_sum
    wheat_share = wheat_sum / total_sum

    if wheat_share > 0.5:
        label = "Wheat Dominant Region"
    elif rice_share > 0.5:
        label = "Rice Dominant Region"
    else:
        label = "Balanced Allocation Region"

    return {
        "label": label,
        "rice_share": rice_share,
        "wheat_share": wheat_share
    }



# -------------------------------------------------------------------
# NATIONAL TREND PLOT
# -------------------------------------------------------------------

def plot_national_allocation(pivot_df):
    """
    Plots national allocation trends for rice, wheat, and total.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Output of get_national_monthly_allocation()

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        pivot_df["date"],
        pivot_df["total_allocated_qty"],
        label="Total Allocation",
        color="black"
    )

    ax.plot(
        pivot_df["date"],
        pivot_df["rice_allocated"],
        label="Rice Allocation",
        color="cyan"
    )

    ax.plot(
        pivot_df["date"],
        pivot_df["wheat_allocated"],
        label="Wheat Allocation",
        color="lightgreen"
    )
    #plt.axvline(pd.Timestamp("2020-03-25"), linestyle="-.", color="red", label="COVID Lockdown")
    ax.set_title("National Monthly Rice & Wheat Allocation (PDS)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocated Quantity (MT)")
    ax.legend()
    ax.grid(True)

    return fig


# -------------------------------------------------------------------
# ZOOMED ANOMALY WINDOW
# -------------------------------------------------------------------

def get_zoomed_period(pivot_df, start_date, end_date):
    """
    Extracts a zoomed time window and computes month-on-month changes.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Output of get_national_monthly_allocation()
    start_date : str or pd.Timestamp
    end_date : str or pd.Timestamp

    Returns
    -------
    pd.DataFrame
        Includes:
        - mom_change
        - mom_pct_change
    """
    period_df = pivot_df[
        (pivot_df["date"] >= start_date) &
        (pivot_df["date"] <= end_date)
    ].copy()

    period_df["mom_change"] = period_df["total_allocated_qty"].diff()
    period_df["mom_pct_change"] = (
        period_df["total_allocated_qty"].pct_change() * 100
    )

    return period_df


# -------------------------------------------------------------------
# ANOMALY PLOT WITH ANNOTATION
# -------------------------------------------------------------------

def plot_allocation_anomaly(period_df):
    """
    Plots zoomed allocation trends and annotates anomalous month
    (largest negative MoM change).

    Parameters
    ----------
    period_df : pd.DataFrame
        Output of get_zoomed_period()

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        period_df["date"],
        period_df["total_allocated_qty"],
        marker="o",
        label="Total",
        color="black"
    )

    ax.plot(
        period_df["date"],
        period_df["rice_allocated"],
        marker="s",
        label="Rice",
        color="cyan"
    )

    ax.plot(
        period_df["date"],
        period_df["wheat_allocated"],
        marker="s",
        label="Wheat",
        color="lightgreen"
    )

    # Identify anomalous month (largest negative MoM)
    collapse_row = period_df.loc[period_df["mom_change"].idxmin()]

    ax.annotate(
        f"Anomaly: Sharp Decline (Under-Reporting)\n({collapse_row['date'].strftime('%b %Y')})",
        xy=(collapse_row["date"], collapse_row["total_allocated_qty"]),
        xytext=(collapse_row["date"], collapse_row["total_allocated_qty"] * 2),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
        ha="center"
    )

    ax.set_title("Allocation Anomaly Diagnostic")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocated Quantity (MT)")
    ax.legend()
    ax.grid(True)

    return fig

# -------------------------------------------------------------------
# ENHANCED NATIONAL TREND WITH ANOMALY WINDOWS
# -------------------------------------------------------------------

def plot_enhanced_national_trends(pivot_df, anomaly_threshold=0.35):

    """
    Plots national allocation trends (Total, Rice, Wheat)
    over full time period, marks anomaly windows based on
    large month-on-month percentage change, and highlights
    COVID lockdown period.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Output of get_national_monthly_allocation()

    anomaly_threshold : float
        Absolute MoM percentage change threshold (default 35%)

    Returns
    -------
    matplotlib.figure.Figure
    """

    df = pivot_df.copy()

    # -------------------------------------------------
    # Level-based anomaly detection using Z-score
    # -------------------------------------------------

    mean_val = df["total_allocated_qty"].mean()
    std_val = df["total_allocated_qty"].std()

    df["z_score"] = (df["total_allocated_qty"] - mean_val) / std_val
    anomaly_dates = df[
        df["z_score"].abs() > 2
    ]["date"]

    initial_window = df[df["date"] < pd.Timestamp("2018-01-01")]["date"]

    anomaly_dates = pd.concat([
        pd.Series(anomaly_dates),
        initial_window
    ]).unique()



    fig, ax = plt.subplots(figsize=(12, 5))

    # -------------------------------------------------
    # Plot main series
    # -------------------------------------------------
    ax.plot(df["date"], df["total_allocated_qty"], label="Total", color="black")
    ax.plot(df["date"], df["rice_allocated"], label="Rice", color="cyan")
    ax.plot(df["date"], df["wheat_allocated"], label="Wheat", color="green")

    # -------------------------------------------------
    # Shade anomaly windows (single neutral color)
    # -------------------------------------------------
    for date in anomaly_dates:
        ax.axvspan(
            date,
            date + pd.DateOffset(months=1),
            color="red",
            alpha=0.12
        )

    # -------------------------------------------------
    # Separate trendlines for Rice and Wheat
    # -------------------------------------------------

    x_numeric = np.arange(len(df))

    # Rice trend
    coef_rice = np.polyfit(x_numeric, df["rice_allocated"].values, 1)
    trend_rice = np.poly1d(coef_rice)

    ax.plot(
        df["date"],
        trend_rice(x_numeric),
        linestyle=":",
        color="cyan",
        linewidth=2,
        label="Rice Trend"
    )

    # Wheat trend
    coef_wheat = np.polyfit(x_numeric, df["wheat_allocated"].values, 1)
    trend_wheat = np.poly1d(coef_wheat)

    ax.plot(
        df["date"],
        trend_wheat(x_numeric),
        linestyle=":",
        color="green",
        linewidth=2,
        label="Wheat Trend"
    )


    # -------------------------------------------------
    # COVID markers
    # -------------------------------------------------
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2020-06-01")

    ax.axvline(covid_start, linestyle="--", color="red", label="Lockdown Start")
    ax.axvline(covid_end, linestyle="--", color="orange", label="Lockdown End")

    # -------------------------------------------------
    # Final styling
    # -------------------------------------------------
    ax.set_title("National Allocation Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocated Quantity (MT)")
    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=90)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    ax.grid(True)  

    return fig,anomaly_dates


def generate_anomaly_reporting_table(df_diag, anomaly_dates):
    """
    Docstring for generate_anomaly_reporting_table
    
    :param df_diag: cleaned dataset
    :param anomaly_dates: dates of anomaly windows
    """

    records = []

    for anomaly_date in anomaly_dates:

        prev_month = anomaly_date - pd.DateOffset(months=1)
        next_month = anomaly_date + pd.DateOffset(months=1)

        for month in [prev_month, anomaly_date, next_month]:

            month_data = df_diag[df_diag["date"] == month]

            num_states = month_data["state_name"].nunique()

            records.append({
                "Month Checked": month,
                "Number of States that Reported Data": num_states
            })

    table_df = pd.DataFrame(records)

    # Remove duplicates (important if months overlap between anomaly windows)
    table_df = table_df.drop_duplicates()

    # Sort ascending by date
    table_df = table_df.sort_values("Month Checked")

    # Format date for display
    table_df["Month Checked"] = table_df["Month Checked"].dt.strftime("%b %Y")

    table_df = table_df.reset_index(drop=True)

    return table_df

def plot_state_commodity_trends(df_diag, selected_state):

    # ---------------------------------------------
    # Filter state-level data
    # ---------------------------------------------
    df_state = df_diag[
        df_diag["state_name"] == selected_state.lower()
    ]

    # ---------------------------------------------
    # Monthly aggregation (state-level)
    # ---------------------------------------------
    monthly_state = (
        df_state.groupby(["date", "commodity"], as_index=False)
        .agg(total_allocated_qty=("total_allocated_qty", "sum"))
    )

    pivot_state = (
        monthly_state
        .pivot(index="date", columns="commodity", values="total_allocated_qty")
        .reindex(columns=["rice", "wheat"])
        .fillna(0)
        .reset_index()
        .rename(columns={
            "rice": "rice_allocated",
            "wheat": "wheat_allocated"
        })
    )

    # ---------------------------------------------
    # Total Allocation for anomaly detection
    # ---------------------------------------------
    pivot_state["total_allocated_qty"] = (
        pivot_state["rice_allocated"] +
        pivot_state["wheat_allocated"]
    )

    # ---------------------------------------------
    # Percentage Change Based Anomaly Detection
    # ---------------------------------------------
    pivot_state["pct_change"] = (
        pivot_state["total_allocated_qty"].pct_change()
    )

    anomaly_dates = pivot_state[
        pivot_state["pct_change"].abs() > 0.35
    ]["date"]


    # ---------------------------------------------
    # Plot
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        pivot_state["date"],
        pivot_state["rice_allocated"],
        label="Rice",
        color="cyan"
    )

    ax.plot(
        pivot_state["date"],
        pivot_state["wheat_allocated"],
        label="Wheat",
        color="green"
    )

    # ---------------------------------------------
    # Shade anomaly windows
    # ---------------------------------------------
    for date in anomaly_dates:
        ax.axvspan(
            date,
            date + pd.DateOffset(months=1),
            color="red",
            alpha=0.12
        )


    # ---------------------------------------------
    # Trendlines
    # ---------------------------------------------
    x_numeric = np.arange(len(pivot_state))

    # Rice trend
    coef_rice = np.polyfit(
        x_numeric,
        pivot_state["rice_allocated"].values,
        1
    )
    trend_rice = np.poly1d(coef_rice)

    ax.plot(
        pivot_state["date"],
        trend_rice(x_numeric),
        linestyle=":",
        color="cyan",
        linewidth=2,
        label="Rice Trend"
    )

    # Wheat trend
    coef_wheat = np.polyfit(
        x_numeric,
        pivot_state["wheat_allocated"].values,
        1
    )
    trend_wheat = np.poly1d(coef_wheat)

    ax.plot(
        pivot_state["date"],
        trend_wheat(x_numeric),
        linestyle=":",
        color="green",
        linewidth=2,
        label="Wheat Trend"
    )

    # ---------------------------------------------
    # Formatting
    # ---------------------------------------------

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(
        mticker.StrMethodFormatter('{x:,.0f}')
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y')
    )

    plt.xticks(rotation=90)

    ax.set_title(f"{selected_state.title()} - Rice & Wheat Allocation Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocated Quantity (MT)")
    ax.grid(True)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig
