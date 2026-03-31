"""
preprocessing.py

Handles:
1. Full timeline creation (per state, commodity)
2. Time-aware imputation (forward fill + backward fill)
3. Outlier clipping (group-wise)

Designed for reuse in:
- Feature engineering notebook
- Forecasting pipeline
"""

import pandas as pd


# -------------------------------------------------------
# 1. Create Full Timeline
# -------------------------------------------------------

def create_full_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each (state_code, commodity) has a complete monthly timeline.

    Parameters:
        df (pd.DataFrame): Input dataframe with columns:
            ['state_code', 'commodity', 'date', 'total_allocated_qty']

    Returns:
        pd.DataFrame: DataFrame with missing months inserted
    """

    def _expand_group(group):
        full_range = pd.date_range(
            start=group['date'].min(),
            end=group['date'].max(),
            freq='M'
        )

        group = group.set_index('date').reindex(full_range)
        group.index.name = 'date'

        # Preserve identifiers
        group['state_code'] = group['state_code'].iloc[0]
        group['commodity'] = group['commodity'].iloc[0]

        return group.reset_index()

    df_full = (
        df.sort_values(['state_code', 'commodity', 'date'])
          .groupby(['state_code', 'commodity'], group_keys=False)
          .apply(_expand_group)
          .reset_index(drop=True)
    )

    return df_full


# -------------------------------------------------------
# 2. Time-aware Imputation
# -------------------------------------------------------

def impute_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies forward fill and backward fill per (state_code, commodity).

    Parameters:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df = df.sort_values(['state_code', 'commodity', 'date'])

    df['total_allocated_qty'] = (
        df.groupby(['state_code', 'commodity'])['total_allocated_qty']
          .ffill()
          .bfill()
    )

    return df


# -------------------------------------------------------
# 3. Outlier Clipping (Group-wise)
# -------------------------------------------------------

def clip_outliers_groupwise(
    df: pd.DataFrame,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95
) -> pd.DataFrame:
    """
    Clips extreme values per (state_code, commodity).

    Parameters:
        df (pd.DataFrame)
        lower_quantile (float): lower bound (default 5%)
        upper_quantile (float): upper bound (default 95%)

    Returns:
        pd.DataFrame
    """

    def _clip(group):
        q_low = group['total_allocated_qty'].quantile(lower_quantile)
        q_high = group['total_allocated_qty'].quantile(upper_quantile)

        group['total_allocated_qty'] = group['total_allocated_qty'].clip(q_low, q_high)
        return group

    df_clipped = (
        df.groupby(['state_code', 'commodity'], group_keys=False)
          .apply(_clip)
          .reset_index(drop=True)
    )

    return df_clipped


# -------------------------------------------------------
# 4. Full Preprocessing Pipeline
# -------------------------------------------------------

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Create full timeline
    2. Impute missing values
    3. Clip outliers

    Parameters:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df = create_full_timeline(df)
    df = clip_outliers_groupwise(df) 
    df = impute_time_series(df)
    
    return df


# -------------------------------------------------------
# 5. Validation Utility (Optional)
# -------------------------------------------------------

def validate_no_missing(df: pd.DataFrame) -> None:
    """
    Prints missing value summary.

    Parameters:
        df (pd.DataFrame)
    """

    missing = df.isna().sum()

    print("Missing values per column:")
    print(missing)

    if missing.sum() == 0:
        print("\n No missing values detected.")
    else:
        print("\n Missing values still present.")