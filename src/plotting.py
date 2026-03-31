
import matplotlib.pyplot as plt


def plot_state_prediction(
    df_sc,
    forecast_dates,
    predictions,
    state,
    commodity
):
    """
    Plots historical actuals + predicted point

    Parameters:
    - df_sc: filtered dataframe for (state, commodity)
    - forecast_dates: prediction dates (datetime)
    - predictions: predicted values 
    """

    df_sc = df_sc.sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 6))

    # --------------------------------------------------
    # Actual trend
    # --------------------------------------------------
    ax.plot(
        df_sc["date"],
        df_sc["total_allocated_qty"],
        label="Actual",
        linewidth=2
    )

    # --------------------------------------------------
    # Highlight prediction point
    # --------------------------------------------------
    actual_points = df_sc[
        df_sc["date"].isin(forecast_dates)
    ]

    if not actual_points.empty:
        ax.scatter(
            actual_points["date"],
            actual_points["total_allocated_qty"],
            s=80,
            label="Actual (Forecast Window)"
        )

    # Predicted point
    ax.scatter(
        forecast_dates,
        predictions,
        s=120,
        marker="X",
        label="Predicted",
    )
    # Connect predictions
    ax.plot(
    forecast_dates,
    predictions,
    linestyle="--",
    alpha=0.7
    )


    # Vertical line for clarity
    ax.axvline(
        x=forecast_dates[0],
        linestyle="--",
        alpha=0.6,
        label="Forecast Start"
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    ax.set_title(f"{state} — {commodity.capitalize()} Allocation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Allocated Quantity")
    ax.set_ylim(bottom=0)

    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=45)

    return fig