# src/date_utils.py
# FOR ALL TIME-RELATED LOGIC - DAYS, DATE, MONTH, YEAR, FINANCIAL YEAR, CALENDAR YEAR

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def extract_calendar_year(year_series):
    """
    Extracts 4-digit calendar year from year_raw column.
    """
    return (
        year_series
        .astype(str)
        .str.extract(r"(\d{4})")[0]
        .astype(int)
    )


def extract_month_name(month_series):
    """
    Extracts month name from strings like 'February, 2021'.
    """
    month_name = (
        month_series
        .str.strip()
        .str.lower()
        .str.extract(
            r"^(january|february|march|april|may|june|july|august|september|october|november|december)"
        )[0]
    )

    assert month_name.notna().all(), "Unparseable month names detected"
    return month_name


def calculate_financial_year_from_month_raw(month_raw_series):
    """
    Calculates Indian financial year (Apr-Mar) from month_raw.
    Returns starting year of the financial year.
    """
    month_name = extract_month_name(month_raw_series)
    c_year = extract_calendar_year(month_raw_series)

    return np.where(
        month_name.isin(["january", "february", "march"]),
        c_year - 1,
        c_year
    )


def create_month_end_date(month_raw_series):
    """
    Creates datetime corresponding to the last day of the month.
    """
    month_name = extract_month_name(month_raw_series)
    c_year = extract_calendar_year(month_raw_series)

    return (
        pd.to_datetime(
            c_year.astype(str) + "-" + month_name + "-01",
            format="%Y-%B-%d",
            errors="coerce"
        )
        + MonthEnd(1)
    )

def create_month_start_date(month_raw_series):
    """
    Creates datetime corresponding to the first day of the month.
    Example:
    - 'April, 2020'    → 2020-04-01
    - 'December, 2018' → 2018-12-01
    """
    month_name = extract_month_name(month_raw_series)
    c_year = extract_calendar_year(month_raw_series)

    return pd.to_datetime(
        c_year.astype(str) + "-" + month_name + "-01",
        format="%Y-%B-%d",
        errors="coerce"
    )


def extract_month_number(date_series):
    """
    Extracts month number (1-12) from a pandas datetime series.
    """
    return date_series.dt.month


def extract_quarter(date_series):
    """
    Extracts calendar quarter (e.g., '2021Q1') from a pandas datetime series.
    """
    return date_series.dt.to_period("Q").astype(str)

def extract_financial_quarter(date_series):
    """
    Extracts Indian financial quarter (Q1 = Apr-Jun).
    """
    return ((date_series.dt.month - 4) % 12 // 3 + 1).astype(int)
