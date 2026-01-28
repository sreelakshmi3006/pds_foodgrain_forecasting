# src/geo_utils.py
# FOR GEOGRAPHIC STANDARDISATION AND CODE MAPPING
# Utilities for geographic standardization and code mapping
# (country, state, district)

import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "additional_data"
DISTRICT_CODES_FILE = ADDITIONAL_DATA_PATH / "District_Names_Codes.xlsx"


# ============================================================
# LOAD MASTER MAPPINGS
# ============================================================

def load_state_mapping():
    """
    Loads state-level mapping:
    state -> state_code, state_num_code
    """
    return pd.read_excel(
        DISTRICT_CODES_FILE,
        sheet_name="state_mapping",
        usecols=["state", "state_code", "state_num_code"]
    )


def load_district_mapping():
    """
    Loads district-level mapping:
    (state_code, district_name) -> district_code, full_district_code
    """
    return pd.read_excel(
        DISTRICT_CODES_FILE,
        sheet_name="district_mapping",
        usecols=[
            "district_name",
            "state_code",
            "district_code",
            "full_district_code"
        ]
    )


# ============================================================
# COUNTRY MAPPING
# ============================================================

def map_country_to_code(series):
    """
    Maps country names to ISO-like country codes.
    Currently supports: india -> IN
    """
    return series.replace({"india": "IN"})


# ============================================================
# STATE MAPPING
# ============================================================

def map_state_codes(df, state_col="state"):
    """
    Adds state_code and state_num_code columns to the DataFrame
    based on cleaned state names.
    """
    state_map = load_state_mapping()

    df = df.merge(
        state_map,
        how="left",
        left_on=state_col,
        right_on="state"
    )

    # Drop duplicate column from mapping table
    df = df.drop(columns=["state"])

    return df


# ============================================================
# DISTRICT PRE-PROCESSING
# ============================================================

def expand_directional_districts(df, district_col="district_name", state_col="state"):
    """
    Expands direction-only district names by appending the state name.
    Example:
        district = 'north', state = 'goa' → 'north goa'

    This only applies when district is EXACTLY one of the directional tokens.
    """
    directional_tokens = {
        "east", "west", "north", "south",
        "north east", "north west",
        "south east", "south west"
    }

    df = df.copy()

    def _expand(row):
        district = row[district_col]
        state = row[state_col]

        if pd.isna(district) or pd.isna(state):
            return district

        if district in directional_tokens:
            return f"{district} {state}"

        return district

    df[district_col] = df.apply(_expand, axis=1)

    return df


# ============================================================
# DISTRICT CODE MAPPING
# ============================================================

def map_district_codes(df, district_col="district_name"):
    """
    Adds district_code and full_district_code columns
    based on (state_code, district_name).

    Assumes:
    - state_code is already present
    - district_col contains cleaned district names
    """
    district_map = load_district_mapping()

    df = df.merge(
        district_map,
        how="left",
        left_on=["state_code", district_col],
        right_on=["state_code", "district_name"]
    )

    # Drop duplicate column from mapping table
    df = df.drop(columns=["district_name"])

    return df
