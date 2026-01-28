# FOR GENERIC STRING CLEANING

import pandas as pd
import re  


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


# FOR NORMALISING DISTRICT NAMES

# -------------------------
# PRECOMPILED REGEX
# -------------------------

# Administrative / office noise (semantic)
ADMIN_RE = re.compile(
    r"\b(ad|ac|dso|dfso|dcso|adc|dcfs|dfsc|idr|sdm)\b,?|"
    r"district food supply office,?|"
    r"district supply office,?|"
    r"district supply section,?|"
    r"district food office,?|"
    r"district food controller,?|"
    r"collector office,?|"
    r"collector office supply branch,?|"
    r"supply branch,?|"
    r"branch supply,?|"
    r"branch food distribution,?|"
    r"district controller,?|"
    r"office of the deputy commissionersupply,?|"
    r"office of the sub divisional officersupply,?|"
    r"food, civil supplies & consumer affairs,?|"
    r"metro,?|"
    r"o/o joint director fcs&ca,?|"
    r"fcs&ca office,?|"
    r"adm supply,?|"
    r"district suppy office,?|"
    r"fcs and ca,?|"
    r"fcs & ca,?|"
    r"fcs&ca,?|"
    r"office name,?|"
    r"o/o the deputy commissioner,?|"
    r"food &civilsupplies,?|"
    r"o/o joint director,?|"
    r"\bdistrict\b,?",
    flags=re.IGNORECASE
)

# CIVIL SUPPLIES prefix (semantic, must come early)
CIVIL_RE = re.compile(
    r"^civil\s+supplies[-\s]*",
    flags=re.IGNORECASE
)

# YSR special case
YSR_RE = re.compile(
    r"\by\.?\s*s\.?\s*r\.?\s*(kadapa)?\b",
    flags=re.IGNORECASE
)

# Punctuation / symbols (character-level)
SYMBOL_RE = re.compile(r"[()\[\]{}._-]")

# -------------------------
# DISTRICT CLEANING FUNCTION
# -------------------------

def normalize_district_name(series: pd.Series) -> pd.Series:
    """
    Cleans and normalizes district names.
    Assumes input is already lowercased.
    """

    s = series.astype(str).str.lower().str.strip()

    # Remove semantic prefixes FIRST
    s = s.str.replace(CIVIL_RE, "", regex=True)
    s = s.str.replace(ADMIN_RE, "", regex=True)

    # Special replacements
    s = s.str.replace(YSR_RE, "kadapa", regex=True)

    # Remove symbols AFTER semantic cleanup
    s = s.str.replace(SYMBOL_RE, " ", regex=True)

    # Final whitespace normalization
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # Restore NaN
    s = s.replace("nan", pd.NA)

    return s
