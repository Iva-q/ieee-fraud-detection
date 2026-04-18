"""Aggregation features by card1 (proxy for client/card).

Each transaction gets per-card1 context: how many transactions this card
has made overall, the typical spending amount, spending variability, etc.

Because these aggregations don't use the target, we can (and should)
compute them over train + test combined — this gives more stable
statistics for cards that appear in both sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_card1_aggregations(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add per-card1 aggregated features to train (and test if provided).

    Aggregations are computed over the concatenation of train and test.
    This is safe (no target leakage) and gives better coverage for rare cards.

    Added columns:
    - `card1_count` : int32, total transactions for this card1 in train+test
    - `card1_amt_mean` : float32, mean TransactionAmt for this card1
    - `card1_amt_std` : float32, std TransactionAmt (NaN if only 1 txn)
    - `card1_amt_max` : float32, max TransactionAmt for this card1
    - `card1_nunique_productcd` : int16, number of distinct ProductCD values
    - `amt_to_card1_mean_ratio` : float32, TransactionAmt / card1_amt_mean

    Parameters
    ----------
    train : pd.DataFrame
        Must contain `card1`, `TransactionAmt`, `ProductCD`.
    test : pd.DataFrame, optional
        Same schema as train (without isFraud). If None, aggregations
        are computed over train only — safe but less stable.

    Returns
    -------
    (train_augmented, test_augmented) : tuple
        Augmented versions of train and test. `test_augmented` is None
        if test was not provided.
    """
    required = ["card1", "TransactionAmt", "ProductCD"]
    for col in required:
        if col not in train.columns:
            raise KeyError(f"add_card1_aggregations requires '{col}' in train")

    # Concat for aggregation — keep an origin flag to split back later
    if test is not None:
        for col in required:
            if col not in test.columns:
                raise KeyError(f"add_card1_aggregations requires '{col}' in test")
        combined = pd.concat(
            [train[required], test[required]],
            axis=0,
            ignore_index=True,
        )
    else:
        combined = train[required].copy()

    # Group once, compute several stats
    grouped = combined.groupby("card1", observed=True, sort=False)
    agg = grouped.agg(
        card1_count=("TransactionAmt", "size"),
        card1_amt_mean=("TransactionAmt", "mean"),
        card1_amt_std=("TransactionAmt", "std"),
        card1_amt_max=("TransactionAmt", "max"),
        card1_nunique_productcd=("ProductCD", "nunique"),
    )

    # Type optimization
    agg["card1_count"] = agg["card1_count"].astype(np.int32)
    agg["card1_amt_mean"] = agg["card1_amt_mean"].astype(np.float32)
    agg["card1_amt_std"] = agg["card1_amt_std"].astype(np.float32)
    agg["card1_amt_max"] = agg["card1_amt_max"].astype(np.float32)
    agg["card1_nunique_productcd"] = agg["card1_nunique_productcd"].astype(np.int16)

    # Merge back to train
    train = train.merge(agg, how="left", left_on="card1", right_index=True)
    train["amt_to_card1_mean_ratio"] = (
        train["TransactionAmt"] / train["card1_amt_mean"]
    ).astype(np.float32)

    # Same for test
    if test is not None:
        test = test.merge(agg, how="left", left_on="card1", right_index=True)
        test["amt_to_card1_mean_ratio"] = (
            test["TransactionAmt"] / test["card1_amt_mean"]
        ).astype(np.float32)

    return train, test