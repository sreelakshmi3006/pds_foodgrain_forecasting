"""
EDA utility functions for PDS Rice & Wheat project.

Reusable aggregation and plotting logic for exploratory analysis.
Framework-agnostic: can be used in notebooks and Streamlit.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
