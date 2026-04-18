"""Money-related features derived from TransactionAmt.

TransactionAmt is the transaction amount in USD. The raw distribution is
highly skewed (long tail of big purchases), so tree models benefit from:
- `log_amt`: compresses the tail, helps model find smoother splits
- `amt_cents`: the decimal part — fraudsters often use "round" numbers
- `amt_has_cents`: binary flag for any non-zero decimal part

These are standard preprocessing choices for payment fraud datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add money-derived features in place.

    Adds:
    - `log_amt` : float32, log(1 + TransactionAmt) — handles the skew
    - `amt_cents` : float32, decimal part of TransactionAmt (0.0..0.99)
    - `amt_has_cents` : int8, 1 if the amount is not a whole number

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `TransactionAmt` column (float).

    Returns
    -------
    pd.DataFrame
        Same dataframe with new columns added.
    """
    if "TransactionAmt" not in df.columns:
        raise KeyError("add_money_features requires 'TransactionAmt' column")

    amt = df["TransactionAmt"].values.astype(np.float64)  # temp precision

    df["log_amt"] = np.log1p(amt).astype(np.float32)

    # Decimal part: 68.50 -> 0.50. Round to avoid float noise.
    cents = np.round(amt - np.floor(amt), 2)
    df["amt_cents"] = cents.astype(np.float32)
    df["amt_has_cents"] = (cents > 0).astype(np.int8)

    return df