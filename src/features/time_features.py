"""Time-based features derived from TransactionDT.

TransactionDT is seconds from an anonymized reference date.
We cannot know the real calendar date, but we can extract:
- hour of day (strong fraud signal in EDA: 10.6% at hour 7 vs 2.3% at hour 13)
- day-of-week (weekend patterns may differ)
- day-from-start (useful for time-aware features like card registration day)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features to the dataframe in place.

    Adds the following columns:
    - `day` : int16, day index from dataset start (TransactionDT // 86400)
    - `hour` : int8, hour of day (0..23) in the reference timezone
    - `dayofweek` : int8, day of week (0..6) assuming day 0 is some weekday
    - `is_night` : int8, binary flag for hour in [0, 10] (high-fraud window from EDA)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a `TransactionDT` column (seconds from dataset start).

    Returns
    -------
    pd.DataFrame
        Same dataframe with new columns added.
    """
    if "TransactionDT" not in df.columns:
        raise KeyError("add_time_features requires 'TransactionDT' column")

    dt = df["TransactionDT"].values

    df["day"] = (dt // 86400).astype(np.int16)
    df["hour"] = ((dt % 86400) // 3600).astype(np.int8)
    df["dayofweek"] = (df["day"] % 7).astype(np.int8)
    df["is_night"] = (df["hour"] <= 10).astype(np.int8)

    return df