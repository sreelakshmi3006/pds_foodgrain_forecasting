import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error
    )

RANDOM_STATE = 42


# --------------------------------------------------
# Utility: Month-end alignment
# --------------------------------------------------
def align_to_month_end(date):
    date = pd.to_datetime(date)
    return date + pd.offsets.MonthEnd(0)


# --------------------------------------------------
# Utility: Compute metrics
# --------------------------------------------------
def compute_metrics(actual, pred):

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = r2_score(actual, pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# --------------------------------------------------
# Core Forecast Function
# --------------------------------------------------
def run_forecast_for_cutoff(
    df,
    year,
    month,
    state,
    commodity,
    feature_cols,
    horizon=3
):

    selected_date = pd.Timestamp(year=year, month=month, day=1)
    selected_date = align_to_month_end(selected_date)

    # --------------------------------------------------
    # Filter state + commodity
    # --------------------------------------------------
    df_sc = df[
        (df["state_code"] == state) &
        (df["commodity"] == commodity)
    ].copy()

    if df_sc.empty:
        return {
            "status": "ERROR",
            "message": f"No data for {commodity} in {state}",
            "predictions": None
        }

    df_sc = df_sc.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # Check if date exists
    # --------------------------------------------------
    if selected_date not in df_sc["date"].values:
        return {
            "status": "ERROR",
            "message": f"No data available for {selected_date.date()}",
            "predictions": None
        }

    idx = df_sc.index[df_sc["date"] == selected_date][0]

    # --------------------------------------------------
    # Check sufficient history (lag_12)
    # --------------------------------------------------
    if idx < 12:
        return {
            "status": "ERROR",
            "message": "Insufficient history (requires at least 12 months)",
            "predictions": None
        }

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------
    train_df = df_sc[df_sc["date"] < selected_date]
    test_row = df_sc[df_sc["date"] == selected_date]

    if train_df.empty or test_row.empty:
        return {
            "status": "ERROR",
            "message": "Invalid train/test split",
            "predictions": None
        }

    X_train = train_df[feature_cols]
    y_train = train_df["total_allocated_qty"]

    X_test = test_row[feature_cols]
    y_test = test_row["total_allocated_qty"].values

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    #PREDICTIONS - RECURSIVE 3 MONTHS
    predictions = []

    df_temp = df_sc.copy()

    current_idx = idx

    for step in range(horizon):

        current_row = df_temp.iloc[current_idx]

        X_input = current_row[feature_cols].values.reshape(1, -1)

        pred = model.predict(X_input)[0]

        predictions.append(pred)

        # Create next row (future timestep)
        next_row = current_row.copy()
        next_row["date"] = current_row["date"] + pd.DateOffset(months=1)
        next_row["total_allocated_qty"] = pred

        # Append
        df_temp = pd.concat([df_temp, pd.DataFrame([next_row])], ignore_index=True)

        # Recompute lag + rolling features

        group_cols = ["state_code", "commodity"]

        df_temp = df_temp.sort_values(["state_code", "commodity", "date"])

        for lag in [1, 2, 3, 6, 9, 12]:
            df_temp[f"lag_{lag}"] = (
                df_temp.groupby(group_cols)["total_allocated_qty"]
                .shift(lag)
            )

        df_temp["rolling_mean_3"] = (
            df_temp.groupby(group_cols)["total_allocated_qty"]
            .transform(lambda x: x.shift(1).rolling(3).mean())
        )

        df_temp["rolling_mean_6"] = (
            df_temp.groupby(group_cols)["total_allocated_qty"]
            .transform(lambda x: x.shift(1).rolling(6).mean())
        )

        df_temp["rolling_std_3"] = (
            df_temp.groupby(group_cols)["total_allocated_qty"]
            .transform(lambda x: x.shift(1).rolling(3).std())
        )

        current_idx += 1

    # --------------------------------------------------
    # Metrics (optional validation window)
    # --------------------------------------------------
    val_df = df_sc.iloc[idx-6:idx]  # last 6 months

    if len(val_df) > 0:
        X_val = val_df[feature_cols]
        y_val = val_df["total_allocated_qty"].values
        val_pred = model.predict(X_val)

        metrics = compute_metrics(y_val, val_pred)
    else:
        metrics = None

    # --------------------------------------------------
    # Anomaly check (volatility proxy)
    # --------------------------------------------------
    recent = df_sc.iloc[idx-12:idx]

    std_ratio = recent["total_allocated_qty"].std() / (
        recent["total_allocated_qty"].mean() + 1e-6
    )

    warning_flag = std_ratio > 0.25

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    if warning_flag:
        return {
            "status": "WARNING",
            "message": "Due to inconsistent data, predictions may be less accurate",
            "predictions": predictions,
            "metrics": metrics,
            "train_df": train_df,
            "val_df": val_df
        }

    return {
        "status": "SUCCESS",
        "message": None,
        "predictions": predictions,
        "metrics": metrics,
        "train_df": train_df,
        "val_df": val_df
    }