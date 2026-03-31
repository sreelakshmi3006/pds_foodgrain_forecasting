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
from datetime import datetime
import plotly.graph_objects as go



# -------------------------------------------------------------------
# YEAR-WISE NATIONAL ALLOCATION (STACKED BAR)
# -------------------------------------------------------------------

def plot_yearly_national_allocation(national_alloc_df):

    """
    Plots year-wise total allocation split by commodity
    using a stacked bar chart.

    Parameters
    ----------
    national_alloc_df : pd.DataFrame
        ['date','commodity','total_allocated_qty']

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    df = national_alloc_df.copy()

    # -------------------------------------------------
    # Extract year
    # -------------------------------------------------

    df["year"] = df["date"].dt.year

    # -------------------------------------------------
    # Aggregate by year and commodity
    # -------------------------------------------------

    yearly_df = (
        df.groupby(["year", "commodity"])["total_allocated_qty"]
        .sum()
        .reset_index()
    )

    # -------------------------------------------------
    # Pivot for stacked bar
    # -------------------------------------------------

    pivot_df = (
        yearly_df.pivot(index="year", columns="commodity", values="total_allocated_qty")
        .fillna(0)
    )

    pivot_df.columns = [str(c).strip().title() for c in pivot_df.columns]

    # Ensure consistent order
    pivot_df = pivot_df[["Rice", "Wheat"]]

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        pivot_df.index,
        pivot_df["Rice"],
        label="Rice",
        color="#8c7a3b"
    )

    ax.bar(
        pivot_df.index,
        pivot_df["Wheat"],
        bottom=pivot_df["Rice"],
        label="Wheat",
        color="#f2c94c"
    )

    # -------------------------------------------------
    # Annotate total allocation above bars
    # -------------------------------------------------

    for year in pivot_df.index:
        total = pivot_df.loc[year].sum()

        ax.text(
            year,
            total * 1.01,
            f"{total/1e6:.1f}M",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # -------------------------------------------------
    # Styling
    # -------------------------------------------------

    ax.set_title("Year-wise National Allocation (Rice vs Wheat)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Allocation (MT)")

    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.legend()

    ax.grid(axis="y")

    plt.tight_layout()

    return fig


# -------------------------------------------------------------------
# MERGE CONTIGUOUS ANOMALY MONTHS
# -------------------------------------------------------------------

def get_contiguous_anomaly_windows(anomaly_dates):

    """
    Converts individual anomaly months into continuous
    anomaly windows.

    Returns
    -------
    List of tuples:
    [(window_start, window_end), ...]
    """

    if len(anomaly_dates) == 0:
        return []

    anomaly_dates = sorted(pd.to_datetime(anomaly_dates))
    windows = []

    start = anomaly_dates[0]
    end = anomaly_dates[0]

    for current in anomaly_dates[1:]:

        # If next anomaly is consecutive month
        if current == end + pd.DateOffset(months=1):
            end = current
        else:
            windows.append((start, end))
            start = current
            end = current

    windows.append((start, end))
    return windows

# -------------------------------------------------------------------
# ENHANCED NATIONAL TREND WITH REPORTING-BASED ANOMALY WINDOWS
# -------------------------------------------------------------------

def plot_enhanced_national_trends(national_alloc_df, state_threshold=24):

    df = national_alloc_df.copy()
    df = df.sort_values("date")

    # -------------------------------------------------
    # Pivot commodity to get rice and wheat series
    # -------------------------------------------------

    pivot_df = (
        df.pivot_table(
            index="date",
            columns="commodity",
            values="total_allocated_qty",
            aggfunc="sum"
        )
        .reset_index()
    )

    pivot_df.columns.name = None

    # Standardize column names (lowercase)
    pivot_df.columns = [str(col).lower() for col in pivot_df.columns]

    # -------------------------------------------------
    # Compute states reporting per date
    # -------------------------------------------------

    states_df = (
        df.groupby("date")["states_reporting"]
        .max()
        .reset_index()
    )

    pivot_df = pivot_df.merge(states_df, on="date", how="left")

    # -------------------------------------------------
    # Detect anomaly windows
    # -------------------------------------------------

    anomaly_mask = pivot_df["states_reporting"] < state_threshold
    anomaly_dates = pivot_df.loc[anomaly_mask, "date"].tolist()

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        pivot_df["date"],
        pivot_df["rice"],
        color="#8c7a3b",
        label="Rice"
    )

    ax.plot(
        pivot_df["date"],
        pivot_df["wheat"],
        color="#f2c94c",
        label="Wheat"
    )

    # -------------------------------------------------
    # Trendlines
    # -------------------------------------------------

    x_numeric = np.arange(len(pivot_df))

    coef_rice = np.polyfit(x_numeric, pivot_df["rice"], 1)
    trend_rice = np.poly1d(coef_rice)

    coef_wheat = np.polyfit(x_numeric, pivot_df["wheat"], 1)
    trend_wheat = np.poly1d(coef_wheat)

    ax.plot(
        pivot_df["date"],
        trend_rice(x_numeric),
        linestyle=":",
        color="#8c7a3b",
        linewidth=2,
        label="Rice Trend"
    )

    ax.plot(
        pivot_df["date"],
        trend_wheat(x_numeric),
        linestyle=":",
        color="#f2c94c",
        linewidth=2,
        label="Wheat Trend"
    )

    # -------------------------------------------------
    # Anomaly windows
    # -------------------------------------------------

    # Merge anomaly months into windows
    windows = get_contiguous_anomaly_windows(anomaly_dates)

    for start, end in windows:

        ax.axvspan(
            start,
            end + pd.DateOffset(months=1),
            color="red",
            alpha=0.15
        )

    # -------------------------------------------------
    # COVID markers
    # -------------------------------------------------

    covid_start = pd.Timestamp("2020-03-25")
    covid_end = pd.Timestamp("2020-06-01")

    ax.axvline(covid_start, linestyle="--", color="red", label="Lockdown Start")
    ax.axvline(covid_end, linestyle="--", color="red", label="Lockdown End")

    # -------------------------------------------------
    # Styling
    # -------------------------------------------------

    ax.set_title("National Allocation Trends (Rice vs Wheat)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocated Quantity (MT)")

    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=90)

    ax.legend(loc="upper left", bbox_to_anchor=(1,1))

    plt.tight_layout(rect=[0,0,0.85,1])

    ax.grid(True)

    return fig, anomaly_dates

# -------------------------------------------------------------------
# NATIONAL ANOMALY REPORTING TABLE
# -------------------------------------------------------------------

def national_anomaly_reporting_table(national_alloc_df, state_threshold=24):

    """
    Returns a table showing months where the number of
    states reporting is below the threshold.

    Parameters
    ----------
    national_alloc_df : pd.DataFrame
        National dataframe with columns:
        ['date','commodity','total_allocated_qty','states_reporting']

    state_threshold : int
        Minimum expected number of reporting states.

    Returns
    -------
    pd.DataFrame
        Columns:
        date, states_reporting
    """

    df = national_alloc_df.copy()

    # Aggregate states reporting per date
    reporting_df = (
        df.groupby("date")["states_reporting"]
        .max()
        .reset_index()
    )

    # Filter anomaly months
    anomaly_df = reporting_df[
        reporting_df["states_reporting"] < state_threshold
    ].copy()

    anomaly_df = anomaly_df.sort_values("date")
    anomaly_df["date"] = anomaly_df["date"].dt.strftime("%b %Y")

    return anomaly_df

# -------------------------------------------------------------------
# COMMODITY DOMINANCE CALCULATOR
# -------------------------------------------------------------------

def commodity_dominance_calculator(df):

    """
    Calculates commodity share distribution and identifies
    the dominant commodity.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing:
        ['commodity', 'total_allocated_qty']

    Returns
    -------
    fig : matplotlib.figure.Figure
        Pie chart of commodity distribution

    dominant_commodity : str
        Commodity with highest allocation share

    dominant_share : float
        Percentage share of dominant commodity
    """

    data = df.copy()

    # -------------------------------------------------
    # Aggregate allocations by commodity
    # -------------------------------------------------

    commodity_totals = (
        data.groupby("commodity")["total_allocated_qty"]
        .sum()
        .reset_index()
    )

    # Clean commodity names
    commodity_totals["commodity_clean"] = (
        commodity_totals["commodity"].str.strip().str.title()
    )

    total_allocation = commodity_totals["total_allocated_qty"].sum()

    commodity_totals["share"] = (
        commodity_totals["total_allocated_qty"] / total_allocation
    )

    # -------------------------------------------------
    # Determine dominant commodity and share
    # -------------------------------------------------

    dominant_row = commodity_totals.loc[
        commodity_totals["share"].idxmax()
    ]

    dominant_commodity = dominant_row["commodity_clean"]
    dominant_share = dominant_row["share"] * 100

    # -------------------------------------------------
    # Plot pie chart
    # -------------------------------------------------

    colors = {
        "Rice": "#8c7a3b",
        "Wheat": "#f2c94c"
    }

    pie_colors = [
        colors.get(c, "#cccccc")
        for c in commodity_totals["commodity_clean"]
    ]

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        commodity_totals["total_allocated_qty"],
        labels=commodity_totals["commodity_clean"],
        autopct="%1.1f%%",
        startangle=90,
        colors=pie_colors,
        wedgeprops={"edgecolor": "white"}
    )

    ax.set_title("Commodity Allocation Distribution")

    plt.tight_layout()

    return fig, dominant_commodity, dominant_share


# -------------------------------------------------------------------
# YEAR-WISE STATE ALLOCATION (STACKED BAR)
# -------------------------------------------------------------------

def plot_yearly_state_allocation(state_df):

    df = state_df.copy()

    df["year"] = df["date"].dt.year

    yearly_df = (
        df.groupby(["year","commodity"])["total_allocated_qty"]
        .sum()
        .reset_index()
    )

    pivot_df = (
        yearly_df.pivot(index="year", columns="commodity", values="total_allocated_qty")
        .fillna(0)
    )

    pivot_df.columns = [str(c).title() for c in pivot_df.columns]

    for col in ["Rice", "Wheat"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df = pivot_df[["Rice", "Wheat"]]

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        pivot_df.index,
        pivot_df["Rice"],
        color="#8c7a3b",
        label="Rice"
    )

    ax.bar(
        pivot_df.index,
        pivot_df["Wheat"],
        bottom=pivot_df["Rice"],
        color="#f2c94c",
        label="Wheat"
    )

    for year in pivot_df.index:
        total = pivot_df.loc[year].sum()
        ax.text(year, total*1.01, f"{total/1e6:.1f}M", ha="center")

    ax.set_title("Year-wise State Allocation")
    ax.set_xlabel("Year")
    ax.set_ylabel("Allocation (MT)")
    ax.legend()
    ax.grid(axis="y")

    plt.tight_layout()

    return fig

# -------------------------------------------------------------------
# BASIC STATE TREND PLOT 
# -------------------------------------------------------------------

def plot_state_allocation(state_df):

    df = state_df.copy()

    pivot_df = (
        df.pivot_table(
            index="date",
            columns="commodity",
            values="total_allocated_qty",
            aggfunc="sum"
        )
        .reset_index()
    )

    pivot_df.columns = [str(c).lower() for c in pivot_df.columns]

    # Ensure both commodities exist
    for col in ["rice", "wheat"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    fig, ax = plt.subplots(figsize=(12,5))

    # Main lines
    ax.plot(pivot_df["date"], pivot_df["rice"], color="#8c7a3b", label="Rice")
    ax.plot(pivot_df["date"], pivot_df["wheat"], color="#f2c94c", label="Wheat")

    x_numeric = np.arange(len(pivot_df))

    # Rice trendline
    if pivot_df["rice"].sum() > 0:
        rice_trend = np.poly1d(np.polyfit(x_numeric, pivot_df["rice"], 1))
        ax.plot(
            pivot_df["date"],
            rice_trend(x_numeric),
            linestyle=":",
            color="#8c7a3b",
            label="Rice Trend"
        )

    # Wheat trendline
    if pivot_df["wheat"].sum() > 0:
        wheat_trend = np.poly1d(np.polyfit(x_numeric, pivot_df["wheat"], 1))
        ax.plot(
            pivot_df["date"],
            wheat_trend(x_numeric),
            linestyle=":",
            color="#f2c94c",
            label="Wheat Trend"
        )

    ax.set_title("State Allocation Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocation (MT)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=90)

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig

# -------------------------------------------------------------------
# ENHANCED STATE TREND WITH ANOMALY WINDOWS
# -------------------------------------------------------------------

def plot_enhanced_state_trends(state_df, anomaly_threshold=0.35):

    df = state_df.copy()

    pivot_df = (
        df.pivot_table(
            index="date",
            columns="commodity",
            values="total_allocated_qty",
            aggfunc="sum"
        )
        .reset_index()
    )

    pivot_df.columns = [str(c).lower() for c in pivot_df.columns]

    for col in ["rice", "wheat"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df["total"] = pivot_df["rice"] + pivot_df["wheat"]

    pivot_df["pct_change"] = pivot_df["total"].pct_change().abs()

    anomaly_dates = pivot_df.loc[
        pivot_df["pct_change"] > anomaly_threshold, "date"
    ].tolist()

    # contiguous anomaly windows
    windows = get_contiguous_anomaly_windows(anomaly_dates)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(pivot_df["date"], pivot_df["rice"], color="#8c7a3b", label="Rice")
    ax.plot(pivot_df["date"], pivot_df["wheat"], color="#f2c94c", label="Wheat")

    x_numeric = np.arange(len(pivot_df))

    if pivot_df["rice"].sum() > 0:
        rice_trend = np.poly1d(np.polyfit(x_numeric, pivot_df["rice"], 1))
        ax.plot(pivot_df["date"], rice_trend(x_numeric),
                linestyle=":", color="#8c7a3b", label="Rice Trend")

    if pivot_df["wheat"].sum() > 0:
        wheat_trend = np.poly1d(np.polyfit(x_numeric, pivot_df["wheat"], 1))
        ax.plot(pivot_df["date"], wheat_trend(x_numeric),
                linestyle=":", color="#f2c94c", label="Wheat Trend")

    # anomaly shading
    for start, end in windows:
        ax.axvspan(start, end + pd.DateOffset(months=1), color="red", alpha=0.15)

    # covid markers
    covid_start = pd.Timestamp("2020-03-25")
    covid_end = pd.Timestamp("2020-06-01")

    ax.axvline(covid_start, linestyle="--", color="red")
    ax.axvline(covid_end, linestyle="--", color="red")

    ax.set_title("Enhanced State Allocation Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocation (MT)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=90)

    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig, anomaly_dates

# -------------------------------------------------------------------
# STATE ANOMALY REPORTING TABLE
# -------------------------------------------------------------------

def state_anomaly_reporting_table(state_df, anomaly_threshold=0.35):

    df = state_df.copy()

    pivot_df = (
        df.pivot_table(
            index="date",
            columns="commodity",
            values="total_allocated_qty",
            aggfunc="sum"
        )
        .reset_index()
    )

    for col in ["rice", "wheat"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    pivot_df.columns = [str(c).lower() for c in pivot_df.columns]

    pivot_df["total"] = pivot_df["rice"] + pivot_df["wheat"]

    pivot_df["pct_change"] = pivot_df["total"].pct_change().abs()

    anomaly_df = pivot_df[pivot_df["pct_change"] > anomaly_threshold].copy()

    anomaly_df["date"] = anomaly_df["date"].dt.strftime("%b %Y")

    table = anomaly_df[
        ["date","rice","wheat"]
    ].rename(columns={
        "date":"Month",
        "rice":"Rice Allocation",
        "wheat":"Wheat Allocation"
    })

    return table.reset_index(drop=True)