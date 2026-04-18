"""Main feature-engineering pipeline for IEEE-CIS.

Orchestrates all feature modules: time, money, aggregations.
Called from notebooks and from training scripts with the same interface,
so that features are identical between experimentation and production.
"""

from __future__ import annotations

import pandas as pd

from src.features.aggregations import add_card1_aggregations
from src.features.encodings import add_frequency_encoding, add_target_encoding
from src.features.money_features import add_money_features
from src.features.time_features import add_time_features
from src.features.uid_features import add_uid, add_uid_aggregations

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    folds: np.ndarray | None = None,
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
        print(f"  [1/6] time features  : +{added} cols -> {train.shape[1]} total")
        n_before = train.shape[1]

    # 2. Money features — also per-row
    train = add_money_features(train)
    if test is not None:
        test = add_money_features(test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [2/6] money features : +{added} cols -> {train.shape[1]} total")
        n_before = train.shape[1]

    # 3. Card1 aggregations — train + test combined for stability
    train, test = add_card1_aggregations(train, test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [3/6] aggregations   : +{added} cols -> {train.shape[1]} total")
        
    # 3b. UID reconstruction + UID aggregations
    train = add_uid(train)
    if test is not None:
        test = add_uid(test)
    train, test = add_uid_aggregations(train, test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [3b/6] UID aggregations: +{added} cols -> {train.shape[1]} total")
        n_before = train.shape[1]
        
    # 4. Frequency encoding — safe, target-independent
    train, test = add_frequency_encoding(train, test)
    if verbose:
        added = train.shape[1] - n_before
        print(f"  [4/6] frequency enc  : +{added} cols -> {train.shape[1]} total")
        n_before = train.shape[1]

    # 5. Target encoding — only if target + folds provided (train-time)
    if target is not None and folds is not None:
        if test is None:
            raise ValueError("Target encoding requires test dataframe")
        train, test = add_target_encoding(train, test, target, folds)
        if verbose:
            added = train.shape[1] - n_before
            print(f"  [5/6] target enc     : +{added} cols -> {train.shape[1]} total")
    else:
        if verbose:
            print("  [5/6] target enc     : skipped (no target/folds provided)")
    if verbose:
        print(f"Done | train shape: {train.shape}")

    return train, test