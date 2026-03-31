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
        usecols=["state", "state_code"]
    )

# ============================================================
# STATE MAPPING
# ============================================================

def map_state_codes(df, state_col="state"):
    """
    Returns a state_code column mapped from state names.
    Does not modify the original DataFrame.
    """

    state_map = load_state_mapping()

    state_dict = dict(
        zip(state_map["state"], state_map["state_code"])
    )
    return df[state_col].map(state_dict)


# -------------------------------------------------------------------
# GET STATE NAMES FROM DATAFRAME
# -------------------------------------------------------------------

def get_state_codes(df, state_code_col="state_code"):
    """
    Returns a dictionary mapping state_code -> state_name
    only for states present in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a state_code column

    state_code_col : str
        Name of the column containing state codes

    Returns
    -------
    dict
        {state_code: state_name}
    """

    # Load mapping table
    state_map = load_state_mapping()

    # Create mapping dictionary
    state_dict = dict(
        zip(state_map["state_code"], state_map["state"])
    )

    # Get unique state codes present in dataframe
    available_codes = sorted(df[state_code_col].dropna().unique())

    # Filter mapping to only those present in dataframe
    filtered_mapping = {
        code: state_dict.get(code, code)
        for code in available_codes
    }

    return filtered_mapping