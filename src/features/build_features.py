"""Main feature-engineering pipeline for IEEE-CIS.

Orchestrates all feature modules: time, money, aggregations.
Called from notebooks and from training scripts with the same interface,
so that features are identical between experimentation and production.
"""

from __future__ import annotations

import pandas as pd

from src.features.aggregations import add_card1_aggregations
from src.features.money_features import add_money_features
from src.features.time_features import add_time_features


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Apply the full feature-engineering pipeline.

    Pipeline steps (order matters for some):
    1. Time features (from TransactionDT): hour, day, dayofweek, is_night
    2. Money features (from TransactionAmt): log_amt, amt_cents, amt_has_cents
    3. Card1 aggregations (over train + test): count, mean, std, max, ratios

    Parameters
    ----------
    train : pd.DataFrame
        Training data. Must contain TransactionDT, TransactionAmt, card1,
        ProductCD.
    test : pd.DataFrame, optional
        Test data. If provided, aggregations use train + test together
        for better stability.
    verbose : bool, default=True
        Print progress after each block.

    Returns
    -------
    (train_features, test_features) : tuple
        Both dataframes augmented with engineered features.
        test_features is None if test was not provided.
    """
    if verbose:
        n_before = train.shape[1]
        print(f"Starting FE pipeline | train shape: {train.shape}")

    # 1. Time features — independent per row, no leakage concerns
    train = add_time_features(train)
    if test is not None:
        test = add_time_features(test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [1/3] time features  : +{added} cols -> {train.shape[1]} total")
        n_before = train.shape[1]

    # 2. Money features — also per-row
    train = add_money_features(train)
    if test is not None:
        test = add_money_features(test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [2/3] money features : +{added} cols -> {train.shape[1]} total")
        n_before = train.shape[1]

    # 3. Card1 aggregations — train + test combined for stability
    train, test = add_card1_aggregations(train, test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [3/3] aggregations   : +{added} cols -> {train.shape[1]} total")

    if verbose:
        print(f"Done | train shape: {train.shape}")

    return train, test