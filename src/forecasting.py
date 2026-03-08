import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FORECAST_HORIZON = 3
RANDOM_STATE = 42

def infer_validation_window(df, forecast_horizon):
    last_valid_date = df["date"].max()
    val_end_date = last_valid_date
    val_start_date = last_valid_date - pd.DateOffset(months=forecast_horizon - 1)
    return val_start_date, val_end_date

def infer_train_end_date(val_start_date, forecast_horizon):
    return val_start_date - pd.DateOffset(months=forecast_horizon)

def make_train_val_split(
    df_state,
    train_start_date,
    train_end_date,
    val_start_date,
    val_end_date
):
    df_state = df_state.sort_values("date")

    train_df = df_state[
        (df_state["date"] >= train_start_date) &
        (df_state["date"] < train_end_date)
    ].reset_index(drop=True)

    val_df = df_state[
        (df_state["date"] >= val_start_date) &
        (df_state["date"] <= val_end_date)
    ].reset_index(drop=True)

    return train_df, val_df


def leakage_check(train_df, val_df, forecast_horizon):
    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation set is empty.")

    latest_train_target_date = (
        train_df["date"] + pd.DateOffset(months=forecast_horizon)
    ).max()

    earliest_val_feature_date = val_df["date"].min()

    if not latest_train_target_date < earliest_val_feature_date:
        raise ValueError("Data leakage detected")


def validate_columns(df, features, target):
    missing = set(features + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def build_pipeline(numeric_features, categorical_features, random_state):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

def train_and_evaluate(
    train_df,
    val_df,
    numeric_features,
    categorical_features,
    target_col,
    random_state,
):
    validate_columns(train_df, numeric_features + categorical_features, target_col)
    validate_columns(val_df, numeric_features + categorical_features, target_col)

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target_col]

    X_val = val_df[numeric_features + categorical_features]
    y_val = val_df[target_col]

    pipeline = build_pipeline(
        numeric_features, categorical_features, random_state
    )

    pipeline.fit(X_train, y_train)

    delta_pred = pipeline.predict(X_val)

    # Reconstructing the predictions
    last_train_y = train_df["total_allocated_qty"].iloc[-1]

    level_preds = []
    prev_level = last_train_y

    for d in delta_pred:
        next_level = prev_level + d
        level_preds.append(next_level)
        prev_level = next_level

    predictions = np.array(level_preds)
    actuals = val_df["total_allocated_qty"].values

    return pipeline, actuals, predictions

def compute_metrics(actuals, predictions, train_df):

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Avoid division by zero in MAPE
    actuals_safe = np.where(actuals == 0, 1, actuals)

    mape = np.mean(
        np.abs((actuals - predictions) / actuals_safe)
    ) * 100

    r2 = r2_score(actuals, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }



def check_lag_coherence(df_state, cutoff, required_lags=3):
    required_dates = [
        cutoff - pd.DateOffset(months=i)
        for i in range(1, required_lags + 1)
    ]

    available_dates = set(df_state["date"])
    missing = [d for d in required_dates if d not in available_dates]

    return len(missing), missing


def run_forecast_for_commodity(
    df,
    commodity,
    state,
    numeric_features,
    categorical_features
):

    # --------------------------------------------
    # Commodity Filtering
    # --------------------------------------------

    # if commodity == "both":
    #     df_state = (
    #         df.groupby(["date"], as_index=False)
    #         .agg(total_allocated_qty=("total_allocated_qty", "sum"))
    #         .sort_values("date")
    #     )
    #     df_state["commodity"] = "both"
        
    # else:

    #     df_state = df[
    #         df["commodity"] == commodity
    #     ].sort_values("date")

    df_state = df[
        (df["commodity"] == commodity) &
        (df["state"] == state)
        ].sort_values("date")

    if df_state.empty:
        return {
            "status": "no_data",
            "message": f"No data available for the selected commodity - {commodity}"
        }
    
    # if commodity == "both":
    #     df_state["commodity"] = "both"


    cutoff = df_state["date"].max()

    # --------------------------------------------
    # Lag coherence check
    # --------------------------------------------
    missing_count, missing_dates = check_lag_coherence(
        df_state, cutoff, required_lags=FORECAST_HORIZON
    )

    if missing_count >= 2:
        return {
            "status": "insufficient_data",
            "message": (
                f"Coherent data not available for {state} "
                f"before {cutoff.date()}. More than one lag month missing."
            )
        }

    imputed = False
    imputed_month = None

    if missing_count == 1:
        imputed = True
        imputed_month = missing_dates[0]

        df_state = df_state.set_index("date").asfreq("MS")
        df_state["total_allocated_qty"] = df_state["total_allocated_qty"].ffill()
        df_state = df_state.reset_index()

    # --------------------------------------------
    # Train/Validation window inference
    # --------------------------------------------
    val_end = cutoff
    val_start = cutoff - pd.DateOffset(months=FORECAST_HORIZON - 1)
    train_end = val_start - pd.DateOffset(months=FORECAST_HORIZON)
    train_start = df_state["date"].min()

    train_df, val_df = make_train_val_split(
        df_state,
        train_start,
        train_end,
        val_start,
        val_end
    )

    if train_df.empty or val_df.empty:
        return {
            "status": "insufficient_data",
            "message": "Train/validation window too small."
        }

    leakage_check(train_df, val_df, FORECAST_HORIZON)

    pipeline, actuals, predictions = train_and_evaluate(
        train_df,
        val_df,
        numeric_features,
        categorical_features,
        target_col="delta_target",
        random_state=RANDOM_STATE,
    )

    metrics = compute_metrics(actuals, predictions, train_df)

    return {
        "status": "success",
        "train_df": train_df,
        "val_df": val_df,
        "predictions": predictions,
        "metrics": metrics,
        "imputed": imputed,
        "imputed_month": imputed_month,
    }


def run_forecast_for_cutoff(
    df,
    year,
    month,
    state,
    commodity,
    numeric_features,
    categorical_features
):
    cutoff = pd.Timestamp(year=year, month=month, day=1)
    df_cut = df[(df["date"] <= cutoff)]
    if df_cut.empty:
        return {
            "status": "no_data",
            "message": "No data available before selected cutoff."
        }

    return run_forecast_for_commodity(
        df_cut,
        commodity,
        numeric_features,
        categorical_features
    )
