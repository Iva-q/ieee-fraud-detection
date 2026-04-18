"""Categorical encodings for IEEE-CIS.

Two families:
1. Frequency encoding — replaces a category value with its count in
   train + test. Safe (target-independent), computed globally.
2. Out-of-fold target encoding — replaces a category value with the
   mean of the target for that category. Must be computed fold-by-fold
   with expanding window to avoid leakage.

Reference: see reports/ideas.md section 4 for the strategy rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Columns to frequency-encode. High-cardinality categoricals and a few
# numeric IDs that behave like categories benefit the most.
FREQUENCY_ENCODE_COLS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
    "uid",
]


def add_frequency_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add frequency-encoded columns for each category in `cols`.

    Frequency is computed over `concat(train, test)` so that rare
    categories get stable counts across both sets. For each column `c`,
    a new column `{c}_freq` is added with the count of occurrences.

    NaN values get NaN in the encoded column too — the model can handle
    this natively. We intentionally do NOT fill NaN with 0, because 0
    would imply "never seen", which conflicts with real NaN semantics
    ("value not provided").

    Parameters
    ----------
    train : pd.DataFrame
        Training data.
    test : pd.DataFrame, optional
        Test data. If provided, its rows participate in the count.
    cols : list of str, optional
        Columns to encode. Defaults to FREQUENCY_ENCODE_COLS.

    Returns
    -------
    (train, test) : tuple
        Augmented dataframes. test is None if not provided.
    """
    if cols is None:
        cols = FREQUENCY_ENCODE_COLS

    # Only encode columns that actually exist
    cols_present = [c for c in cols if c in train.columns]
    missing = set(cols) - set(cols_present)
    if missing:
        print(f"[frequency_encoding] Skipping missing columns: {sorted(missing)}")

    for col in cols_present:
        # Concatenate train[col] and test[col] if test is given
        if test is not None and col in test.columns:
            combined = pd.concat([train[col], test[col]], axis=0, ignore_index=True)
        else:
            combined = train[col]

        # value_counts ignores NaN by default, which is what we want
        counts = combined.value_counts(dropna=True)

        # Map back to original columns. Rows where the value is NaN
        # (or never seen) will get NaN in the new column.
        new_col = f"{col}_freq"
        train[new_col] = train[col].map(counts).astype("float32")
        if test is not None and col in test.columns:
            test[new_col] = test[col].map(counts).astype("float32")

    return train, test



    # Columns to target-encode. Fewer than frequency-encoded because target
# encoding is more expensive (recomputed per fold) and can overfit on
# high-cardinality columns with few-sample categories.
TARGET_ENCODE_COLS = [
    "card1",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "ProductCD",
    "DeviceInfo",
    "uid",
]


def _expanding_target_encode(
    train_values: pd.Series,
    target: pd.Series,
    folds: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Compute out-of-fold expanding-window target encoding for a single column.

    For each fold i:
        - Use only rows with fold < i (past) to compute category means
        - Apply those means to rows with fold == i
    For fold 0, which has no past, fill with the global mean from fold 0 itself
    (this is the bootstrap; fold 0 is not used for model validation anyway).

    Smoothing blends the per-category mean with the global mean,
    weighted by how many samples the category has:
        encoded = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
    Large `smoothing` shrinks rare categories toward the global mean.

    Parameters
    ----------
    train_values : pd.Series
        Categorical column to encode.
    target : pd.Series
        Binary target (isFraud).
    folds : np.ndarray
        Fold assignments from make_time_folds.
    smoothing : float
        Smoothing strength. Values 5-50 are typical.

    Return
    -------
    np.ndarray of shape (len(train_values),)
        Target-encoded values, one per row.
    """
    encoded = np.full(len(train_values), np.nan, dtype=np.float32)
    n_folds = int(folds.max()) + 1

    global_mean_full = target.mean()  # fallback для fold 0

    for fold in range(n_folds):
        valid_mask = folds == fold

        if fold == 0:
            # Fold 0 has no past — fill with global mean from the entire train.
            # This makes the TE feature constant on fold 0 (useless but safe).
            # Using fold 0 itself would create distribution shift between
            # train (folds 0-N) and validation (fold 1+), hurting the model.
            encoded[valid_mask] = np.float32(global_mean_full)
            continue

        past_mask = folds < fold
        past_values = train_values.loc[past_mask]
        past_target = target.loc[past_mask]
        global_mean = past_target.mean()

        # Per-category stats on the "past" slice
        stats = past_target.groupby(past_values, observed=True).agg(["mean", "size"])
        smoothed = (
            stats["size"] * stats["mean"] + smoothing * global_mean
        ) / (stats["size"] + smoothing)

        # Apply to valid fold
        valid_values = train_values.loc[valid_mask]
        # map() preserves the source dtype (e.g. categorical) — force float
        encoded_values = valid_values.map(smoothed).astype("float32")
        # Categories unseen in the past → global mean
        encoded_values = encoded_values.fillna(np.float32(global_mean))

        encoded[valid_mask] = encoded_values.values

    return encoded


def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    folds: np.ndarray,
    cols: list[str] | None = None,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add out-of-fold target encoding to train and test.

    Train encoding uses expanding-window OOF scheme to prevent leakage.
    Test encoding uses the full train (no OOF needed — target is unknown
    for test rows, so there's nothing to leak).

    Parameters
    ----------
    train : pd.DataFrame
        Training data (will be augmented with `{col}_te` columns).
    test : pd.DataFrame
        Test data (will be augmented with the same columns).
    target : pd.Series
        Binary target aligned with train rows.
    folds : np.ndarray
        Fold assignments for train, same length as train.
    cols : list of str, optional
        Columns to encode. Defaults to TARGET_ENCODE_COLS.
    smoothing : float, default=10
        Bayesian smoothing strength.

    Returns
    -------
    (train, test) : tuple
        Augmented dataframes.
    """
    if cols is None:
        cols = TARGET_ENCODE_COLS
    cols_present = [c for c in cols if c in train.columns and c in test.columns]
    missing = set(cols) - set(cols_present)
    if missing:
        print(f"[target_encoding] Skipping missing columns: {sorted(missing)}")

    global_mean = target.mean()

    for col in cols_present:
        new_col = f"{col}_te"

        # OOF encoding on train
        train[new_col] = _expanding_target_encode(
            train_values=train[col],
            target=target,
            folds=folds,
            smoothing=smoothing,
        )

        # Test encoding: use the full train to compute category means
        stats = target.groupby(train[col], observed=True).agg(["mean", "size"])
        smoothed = (
            stats["size"] * stats["mean"] + smoothing * global_mean
        ) / (stats["size"] + smoothing)

        test_encoded = test[col].map(smoothed).astype("float32").fillna(np.float32(global_mean))
        test[new_col] = test_encoded.values

    return train, test