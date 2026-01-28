import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FORECAST_HORIZON = 3
TRAIN_START_DATE = pd.Timestamp("2018-04-01")
RANDOM_STATE = 42

def infer_validation_window(df, forecast_horizon):
    last_valid_date = df["date"].max()
    val_end_date = last_valid_date
    val_start_date = last_valid_date - pd.DateOffset(months=forecast_horizon - 1)
    return val_start_date, val_end_date

def infer_train_end_date(val_start_date, forecast_horizon):
    return val_start_date - pd.DateOffset(months=forecast_horizon)

def make_train_val_split(
    df,
    commodity,
    train_start_date,
    train_end_date,
    val_start_date,
    val_end_date
):
    df_c = df[df["commodity"] == commodity].sort_values(["state", "date"])

    train_df = df_c[
        (df_c["date"] >= train_start_date) &
        (df_c["date"] < train_end_date)
    ].reset_index(drop=True)

    val_df = df_c[
        (df_c["date"] >= val_start_date) &
        (df_c["date"] <= val_end_date)
    ].reset_index(drop=True)

    return train_df, val_df

def leakage_check(train_df, val_df, forecast_horizon):
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

    level_pred = val_df["total_allocated_qty"].values + delta_pred
    level_true = val_df["target"].values

    return pipeline, level_true, level_pred

def compute_metrics(level_true, level_pred, train_df):
    mae = mean_absolute_error(level_true, level_pred)
    rmse = np.sqrt(mean_squared_error(level_true, level_pred))
    r2 = r2_score(level_true, level_pred)

    mean_train = train_df["target"].mean()
    std_train = train_df["target"].std()

    return {
        "R2": r2,
        "nMAE": mae / mean_train,
        "nRMSE": rmse / std_train,
    }

def run_forecast_for_commodity(df, commodity, numeric_features, categorical_features):
    val_start, val_end = infer_validation_window(df, FORECAST_HORIZON)
    train_end = infer_train_end_date(val_start, FORECAST_HORIZON)

    train_df, val_df = make_train_val_split(
        df,
        commodity,
        TRAIN_START_DATE,
        train_end,
        val_start,
        val_end,
    )

    leakage_check(train_df, val_df, FORECAST_HORIZON)

    pipeline, level_true, level_pred = train_and_evaluate(
        train_df,
        val_df,
        numeric_features,
        categorical_features,
        target_col="delta_target",
        random_state=RANDOM_STATE,
    )

    metrics = compute_metrics(level_true, level_pred, train_df)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "level_true": level_true,
        "level_pred": level_pred,
        "metrics": metrics,
    }

def run_forecast_for_cutoff(df, year, month, numeric_features, categorical_features):
    cutoff = pd.Timestamp(year=year, month=month, day=1)
    df_cut = df[df["date"] <= cutoff]

    rice = run_forecast_for_commodity(
        df_cut, "rice", numeric_features, categorical_features
    )
    wheat = run_forecast_for_commodity(
        df_cut, "wheat", numeric_features, categorical_features
    )

    return {"rice": rice, "wheat": wheat}