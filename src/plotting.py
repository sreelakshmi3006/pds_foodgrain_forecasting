import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import matplotlib.dates as mdates



def plot_actual_vs_predicted(train_df, val_df, level_pred, commodity):

    actuals = pd.concat(
        [train_df[["date", "target"]], val_df[["date", "target"]]]
    )

    actuals_monthly = (
        actuals.groupby("date")["target"]
        .sum()
        .reset_index()
    )

    preds_monthly = (
        val_df.assign(predicted=level_pred)
        .groupby("date")["predicted"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        actuals_monthly["date"],
        actuals_monthly["target"],
        label="Actual",
    )

    ax.plot(
        preds_monthly["date"],
        preds_monthly["predicted"],
        label="Predicted",
        marker="o",
    )

    ax.axvline(
        x=preds_monthly["date"].min(),
        linestyle=":",
        color="grey",
        label="Forecast Start",
    )

    # 🔹 FORMAT X AXIS
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
    plt.xticks(rotation=90)

    ax.set_title(f"{commodity.capitalize()} — Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Allocated Quantity")
    ax.legend()
    ax.grid(True)

    return fig


