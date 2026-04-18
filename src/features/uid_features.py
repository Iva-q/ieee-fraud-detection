"""UID-based features — the signature technique of top IEEE-CIS solutions.

The dataset has no explicit client ID (Vesta anonymized it). But we can
reconstruct a proxy UID from three columns:
    UID = card1 + addr1 + D1n
where D1n = D1 - day (the day the card was registered, constant per client).

This UID is NOT unique per person — it's a group that shares card BIN,
address region, and card registration day. For clients with few cards, it
behaves as a tight client ID. For busy BINs (e.g. a popular Chase BIN in a
large city), it may still group multiple clients together, but much less
than card1 alone (which had 28K-size groups in our EDA).

Reference: see reports/ideas.md section 7.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Add D1n and UID columns to the dataframe.

    D1n = D1 - day (day of card registration relative to dataset start)
    UID = string concatenation of card1, addr1, and D1n

    Rows with missing D1 or addr1 will have NaN in UID — downstream code
    must handle this gracefully (pandas treats NaN as distinct in groupby,
    so we convert to a sentinel string below).

    Requires columns: card1, addr1, D1, TransactionDT
    (day will be computed from TransactionDT if absent)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Same dataframe with 'D1n' and 'uid' columns added.
    """
    for col in ("card1", "addr1", "D1", "TransactionDT"):
        if col not in df.columns:
            raise KeyError(f"add_uid requires '{col}' in dataframe")

    # Compute day if not present
    if "day" not in df.columns:
        df["day"] = (df["TransactionDT"] // 86400).astype(np.int16)

    # D1n: card registration day relative to dataset start (constant per client)
    df["D1n"] = (df["D1"] - df["day"]).astype("float32")

    # UID as a string concatenation. Using strings (not hashes) keeps the
    # feature human-readable for debugging; the pd.Categorical wrapper
    # later will make groupby fast.
    # Missing components -> "NA" sentinel, so the resulting UID is only
    # valid when all three parts are present.
    card1_str = df["card1"].astype("Int32").astype("string").fillna("NA")
    addr1_str = df["addr1"].astype("Int32").astype("string").fillna("NA")
    d1n_str = df["D1n"].astype("Int32").astype("string").fillna("NA")

    uid = card1_str + "_" + addr1_str + "_" + d1n_str

    # Mark UIDs with any missing component as NaN (they shouldn't produce
    # real statistics — there's no real client behind them)
    has_all = (
        df["card1"].notna()
        & df["addr1"].notna()
        & df["D1"].notna()
    )
    uid = uid.where(has_all, other=pd.NA)

    df["uid"] = uid.astype("category")

    return df


def add_uid_aggregations(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add per-UID aggregated features, computed on train + test together.

    This mirrors add_card1_aggregations but uses the reconstructed UID.
    UID-based aggregations are much stronger because they identify groups
    of 1-10 transactions (real clients), not 1000-28000 (card1 BINs).

    Added columns:
    - uid_count : int32, total txns for this UID
    - uid_amt_mean, uid_amt_std, uid_amt_max : float32
    - uid_nunique_productcd : int16
    - amt_to_uid_mean_ratio : float32, current txn amt / uid's mean amt

    Rows where UID is NaN (missing card1/addr1/D1) get NaN in all these
    columns — LightGBM handles it natively.

    Parameters
    ----------
    train : pd.DataFrame
        Must have `uid`, `TransactionAmt`, `ProductCD`.
    test : pd.DataFrame, optional
        Same schema.

    Returns
    -------
    (train_augmented, test_augmented) : tuple
    """
    required = ["uid", "TransactionAmt", "ProductCD"]
    for col in required:
        if col not in train.columns:
            raise KeyError(f"add_uid_aggregations requires '{col}' in train")

    if test is not None:
        combined = pd.concat(
            [train[required], test[required]],
            axis=0, ignore_index=True,
        )
    else:
        combined = train[required].copy()

    # Group by UID (categorical), observed=True ignores unused categories
    grouped = combined.groupby("uid", observed=True, sort=False)
    agg = grouped.agg(
        uid_count=("TransactionAmt", "size"),
        uid_amt_mean=("TransactionAmt", "mean"),
        uid_amt_std=("TransactionAmt", "std"),
        uid_amt_max=("TransactionAmt", "max"),
        uid_nunique_productcd=("ProductCD", "nunique"),
    )

    agg["uid_count"] = agg["uid_count"].astype("float32")  # float to allow NaN
    agg["uid_amt_mean"] = agg["uid_amt_mean"].astype("float32")
    agg["uid_amt_std"] = agg["uid_amt_std"].astype("float32")
    agg["uid_amt_max"] = agg["uid_amt_max"].astype("float32")
    agg["uid_nunique_productcd"] = agg["uid_nunique_productcd"].astype("float32")

    # Merge back — UIDs that are NaN won't match anything, so those rows
    # naturally get NaN in the new columns.
    train = train.merge(agg, how="left", left_on="uid", right_index=True)
    train["amt_to_uid_mean_ratio"] = (
        train["TransactionAmt"] / train["uid_amt_mean"]
    ).astype("float32")

    if test is not None:
        test = test.merge(agg, how="left", left_on="uid", right_index=True)
        test["amt_to_uid_mean_ratio"] = (
            test["TransactionAmt"] / test["uid_amt_mean"]
        ).astype("float32")

    return train, test