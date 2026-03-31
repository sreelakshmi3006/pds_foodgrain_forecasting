# FOR GENERIC STRING CLEANING

import pandas as pd
import re  

def is_zero(df):
    return df == 0

def standardise_columns(df):
    """
    Standardizes column names:
    - strips whitespace
    - converts to lowercase
    - replaces spaces with underscores
    - removes special characters
    - normalizes underscores
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)              # spaces → underscore
        .str.replace(r"[^a-z0-9_]", "", regex=True)        # remove special chars
        .str.replace(r"_+", "_", regex=True)               # collapse underscores
        .str.strip("_")                                    # trim underscores
    )

    return df

def normalize_text_column(series):
    """
    Normalizes text data by:
    - converting to lowercase
    - stripping leading/trailing whitespace
    - replacing brackets, dots, hyphens, underscores with spaces
    - collapsing multiple spaces
    - safely handling NaN values
    """
    return (
        series
        .astype(str)
        .str.lower()
        .str.replace(r"[()\[\]{}._-]", " ", regex=True)  # replace with space
        .str.replace(r"\s+", " ", regex=True)            # collapse spaces
        .str.strip()
        .replace("nan", pd.NA)
    )


def normalize_indian_state_names(series):
    """
    Normalizes known Indian state / UT naming variations
    to a canonical form and removes leading 'the'.
    """
    # Explicit domain mappings
    state_mapping = {
        "daman & diu": "dadra and nagar haveli and daman and diu",
        "dadra & nagar haveli": "dadra and nagar haveli and daman and diu",
        "dadar & nagar haveli": "dadra and nagar haveli and daman and diu",
        "jammu & kashmir": "jammu and kashmir"
    }

    series = series.replace(state_mapping)

    # Remove leading 'the ' if present
    series = series.str.replace(r"^the\s+", "", regex=True)

    return series
