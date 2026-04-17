"""Time-based cross-validation splits for IEEE-CIS Fraud Detection.

The dataset spans 182 days with shifting fraud rate over time.
Random KFold leaks future information into training folds — we must use
time-based splits where validation always follows training chronologically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_time_folds(
    time_values: pd.Series | np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Assign each row to a time-based fold.

    Rows are sorted by `time_values` and split into `n_splits` equal-sized
    consecutive chunks. The returned array has the same index as the input
    and contains fold numbers 0..n_splits-1, where fold 0 is the earliest.

    Parameters
    ----------
    time_values : pd.Series or np.ndarray
        Timestamps (e.g., TransactionDT). Values are compared directly,
        so any monotonic time representation works.
    n_splits : int, default=5
        Number of folds.

    Returns
    -------
    np.ndarray of shape (n_rows,), dtype=int8
        Fold assignment for each row, preserving original row order.

    Examples
    --------
    >>> folds = make_time_folds(train["TransactionDT"], n_splits=5)
    >>> # Rows with smallest TransactionDT get fold 0, largest get fold 4.
    """
    time_arr = np.asarray(time_values)
    n = len(time_arr)

    # Indices that would sort time_values ascending
    sort_idx = np.argsort(time_arr, kind="stable")

    # Assign consecutive chunks to consecutive folds
    fold_of_sorted = np.empty(n, dtype=np.int8)
    chunk_size = n // n_splits
    for fold in range(n_splits):
        start = fold * chunk_size
        end = (fold + 1) * chunk_size if fold < n_splits - 1 else n
        fold_of_sorted[start:end] = fold

    # Map back to original row order
    folds = np.empty(n, dtype=np.int8)
    folds[sort_idx] = fold_of_sorted
    return folds


def expanding_window_splits(
    folds: np.ndarray,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window (train, valid) index pairs.

    For fold i in [1..n_splits-1]:
        train = all rows with fold < i
        valid = rows with fold == i
    Fold 0 is never used as validation (no past data to train on).

    This mimics real production: at prediction time t, we can only use
    transactions that happened before t.

    Parameters
    ----------
    folds : np.ndarray
        Output of `make_time_folds`.
    n_splits : int, default=5
        Number of folds (must match the one used to build `folds`).

    Returns
    -------
    list of (train_idx, valid_idx) tuples
        Each element is a pair of integer arrays indexing into the original
        dataset.

    Examples
    --------
    >>> folds = make_time_folds(train["TransactionDT"], n_splits=5)
    >>> for train_idx, valid_idx in expanding_window_splits(folds):
    ...     X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    ...     X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]
    ...     # train model, evaluate on valid
    """
    splits = []
    for fold in range(1, n_splits):
        train_idx = np.where(folds < fold)[0]
        valid_idx = np.where(folds == fold)[0]
        splits.append((train_idx, valid_idx))
    return splits