import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from src.eda_utils import (
    get_national_monthly_allocation,
    plot_national_allocation,
    get_zoomed_period,
    plot_allocation_anomaly
)

def render_national_trends(df):
    """
    Renders national allocation trends plot.
    """
    pivot_df = get_national_monthly_allocation(df)
    fig = plot_national_allocation(pivot_df)
    return fig

def render_anomaly_diagnostics(
    df,
    start_date="2020-07-01",
    end_date="2021-01-31"
):
    """
    Renders anomaly diagnostic plot for a given period.
    """
    pivot_df = get_national_monthly_allocation(df)

    period_df = get_zoomed_period(
        pivot_df,
        start_date=start_date,
        end_date=end_date
    )

    fig = plot_allocation_anomaly(period_df)
    return fig


def plot_national_trends(df):
    monthly = (
        df.groupby("date")["total_allocated_qty"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["date"], monthly["total_allocated_qty"])
    ax.set_title("National PDS Allocation Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Allocated Quantity")
    ax.grid(True)

    return fig

def plot_anomaly_diagnostics(df):
    monthly = (
        df.groupby(["date", "is_anomaly"])["total_allocated_qty"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for flag, g in monthly.groupby("is_anomaly"):
        label = "Anomaly" if flag == 1 else "Normal"
        ax.plot(g["date"], g["total_allocated_qty"], label=label)

    ax.set_title("Anomaly Diagnostics")
    ax.legend()
    ax.grid(True)

    return fig
