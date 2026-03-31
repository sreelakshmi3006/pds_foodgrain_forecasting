# src/date_utils.py
# FOR ALL TIME-RELATED LOGIC - DAYS, DATE, MONTH, YEAR, FINANCIAL YEAR, CALENDAR YEAR

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

def extract_date_from_year_and_month(year_raw_series, month_raw_series):
    """
    Creates a pandas datetime column representing the last day of the month
    using year_raw and month_raw columns.

    Parameters
    ----------
    year_raw_series : pd.Series
        Series containing values like 'Calendar Year (Jan - Dec), 2021'

    month_raw_series : pd.Series
        Series containing values like 'September, 2021'

    Returns
    -------
    pd.Series
        Datetime series where each value is the last day of the corresponding month.
    """

    # Extract numeric year
    year = year_raw_series.str.extract(r'(\d{4})').astype(int)[0]

    # Extract month name
    month_name = month_raw_series.str.extract(r'([A-Za-z]+)')[0].str.title()

    # Convert month name to month number
    month_num = pd.to_datetime(month_name, format="%B").dt.month

    # Create datetime (first day of month)
    date = pd.to_datetime({
        "year": year,
        "month": month_num,
        "day": 1
    })

    # Shift to last day of month
    date = date + MonthEnd(1)

    return date


def extract_month_number(date_series):
    """
    Extracts the month number (1-12) from a datetime series.

    Example:
    2022-03-31 → 3
    """
    return date_series.dt.month