"""Inference-time preprocessor for the fraud detection model.

Reimplements the feature engineering from src/features/ in a way suitable
for single-transaction inference:

- During fit(): runs the existing FE pipeline on train+test, extracts all
  lookup tables (frequency, target encoding, UID aggregations).
- During transform(): takes a raw transaction dict, applies the same
  transformations using the saved lookups (no groupby, no TE recomputation).

The class is picklable, so we save it once after training and load at
service startup.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_features import build_features
from src.features.encodings import FREQUENCY_ENCODE_COLS, TARGET_ENCODE_COLS


class FraudPreprocessor:
    """Stateful preprocessor that reproduces our FE at inference time.

    Usage
    -----
    >>> pre = FraudPreprocessor()
    >>> pre.fit(train, test, target, folds, selected_features)
    >>> pre.save("models/preprocessor.pkl")
    >>> # later, in the service:
    >>> pre = FraudPreprocessor.load("models/preprocessor.pkl")
    >>> features = pre.transform(raw_transaction_dict)
    >>> prediction = model.predict(features)
    """

    def __init__(self) -> None:
        # Column lists
        self.selected_features: list[str] = []
        self.cat_cols: list[str] = []
        # dtype map for categorical columns (needed so LGBM sees the same
        # category universe as during training)
        self.cat_categories: dict[str, list] = {}
        # dtype map for non-categorical columns (to restore types after JSON round-trip)
        self.numeric_dtypes: dict[str, str] = {}
        # Lookups
        self.freq_lookups: dict[str, dict] = {}
        self.te_lookups: dict[str, dict] = {}
        self.uid_agg_lookup: dict[str, dict] = {}
        self.card1_agg_lookup: dict[int, dict] = {}
        # Global stats for fallbacks
        self.global_target_mean: float = 0.0
        # Dataset-level anchor for computing `day` at inference
        # (day = TransactionDT // 86400)
        # No state needed — stateless calc
        # Health
        self._fitted: bool = False

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: pd.Series,
        folds: np.ndarray,
        selected_features: list[str],
    ) -> "FraudPreprocessor":
        """Build all lookup tables from train + test.

        This runs the full FE pipeline once to get the fully-featured
        train and test, then extracts per-column lookups from them.
        """
        # 1) Run the normal FE pipeline to get everything computed once
        train_fe, test_fe = build_features(
            train=train.copy(),
            test=test.copy(),
            target=target,
            folds=folds,
            verbose=False,
        )

        combined = pd.concat([train_fe, test_fe], ignore_index=True)

        # 2) Save column lists
        self.selected_features = list(selected_features)

        # 3) Save dtype info
        # - For categoricals: we need the full category universe so LGBM
        #   at inference sees the same set as training.
        # - For numerics: we need the exact dtype to restore it after
        #   JSON round-trip (JSON kills numeric typing for missing values).
        for col in selected_features:
            if col not in train_fe.columns:
                continue
            s = train_fe[col]
            if isinstance(s.dtype, pd.CategoricalDtype):
                self.cat_cols.append(col)
                self.cat_categories[col] = list(s.cat.categories)
            else:
                self.numeric_dtypes[col] = str(s.dtype)

        # 4) Frequency lookups
        for col in FREQUENCY_ENCODE_COLS:
            if col not in combined.columns:
                continue
            counts = combined[col].value_counts(dropna=True)
            # Convert keys to python scalars for picklability / JSON-safety
            self.freq_lookups[col] = {
                (k if not pd.isna(k) else None): int(v) for k, v in counts.items()
            }

        # 5) Target encoding lookups — using the FULL train
        # (we already did OOF inside train_fe; for inference we need a
        # plain per-category mean from all train rows)
        self.global_target_mean = float(target.mean())
        smoothing = 10.0  # must match add_target_encoding default

        for col in TARGET_ENCODE_COLS:
            if col not in train.columns and col != "uid":
                continue
            # uid is computed during FE — we need to grab it from train_fe
            source = train_fe[col] if col in train_fe.columns else None
            if source is None:
                continue

            stats = target.groupby(source, observed=True).agg(["mean", "size"])
            smoothed = (
                stats["size"] * stats["mean"] + smoothing * self.global_target_mean
            ) / (stats["size"] + smoothing)
            self.te_lookups[col] = {
                (k if not pd.isna(k) else None): float(v) for k, v in smoothed.items()
            }

        # 6) UID aggregation lookup
        uid_grouped = combined.groupby("uid", observed=True).agg(
            uid_count=("TransactionAmt", "size"),
            uid_amt_mean=("TransactionAmt", "mean"),
            uid_amt_std=("TransactionAmt", "std"),
            uid_amt_max=("TransactionAmt", "max"),
            uid_nunique_productcd=("ProductCD", "nunique"),
        )
        self.uid_agg_lookup = {
            str(k): {
                "uid_count": float(row["uid_count"]),
                "uid_amt_mean": float(row["uid_amt_mean"]),
                "uid_amt_std": float(row["uid_amt_std"])
                if not pd.isna(row["uid_amt_std"])
                else None,
                "uid_amt_max": float(row["uid_amt_max"]),
                "uid_nunique_productcd": float(row["uid_nunique_productcd"]),
            }
            for k, row in uid_grouped.iterrows()
        }

        # 7) Card1 aggregation lookup
        card1_grouped = combined.groupby("card1", observed=True).agg(
            card1_count=("TransactionAmt", "size"),
            card1_amt_mean=("TransactionAmt", "mean"),
            card1_amt_std=("TransactionAmt", "std"),
            card1_amt_max=("TransactionAmt", "max"),
            card1_nunique_productcd=("ProductCD", "nunique"),
        )
        self.card1_agg_lookup = {
            int(k): {
                "card1_count": float(row["card1_count"]),
                "card1_amt_mean": float(row["card1_amt_mean"]),
                "card1_amt_std": float(row["card1_amt_std"])
                if not pd.isna(row["card1_amt_std"])
                else None,
                "card1_amt_max": float(row["card1_amt_max"]),
                "card1_nunique_productcd": float(row["card1_nunique_productcd"]),
            }
            for k, row in card1_grouped.iterrows()
        }

        self._fitted = True
        return self

    # ------------------------------------------------------------- transform

    def transform(self, raw: dict[str, Any]) -> pd.DataFrame:
        """Convert a single raw transaction dict into the 261-feature frame.

        Parameters
        ----------
        raw : dict
            Raw transaction. Must contain at least TransactionDT,
            TransactionAmt, card1, addr1, D1, ProductCD (plus whatever
            other columns end up being selected).

        Returns
        -------
        pd.DataFrame
            One-row dataframe with `self.selected_features` as columns,
            correct dtypes, ready for model.predict().
        """
        if not self._fitted:
            raise RuntimeError("FraudPreprocessor is not fitted")

        # Start from a copy of the raw dict (so we don't mutate input)
        row = dict(raw)

        # --- 1. Time features
        dt = float(row.get("TransactionDT", 0))
        day = int(dt // 86400)
        row["day"] = day
        row["hour"] = int((dt // 3600) % 24)
        row["dayofweek"] = day % 7
        row["is_night"] = int(row["hour"] <= 10)

        # --- 2. Money features
        amt = float(row.get("TransactionAmt", 0))
        row["log_amt"] = float(np.log1p(amt))
        cents = amt - int(amt)
        row["amt_cents"] = float(cents)
        row["amt_has_cents"] = int(cents > 1e-6)

        # --- 3. Card1 aggregations from lookup
        card1 = row.get("card1")
        card1_stats = self.card1_agg_lookup.get(int(card1)) if pd.notna(card1) else None
        if card1_stats:
            for k, v in card1_stats.items():
                row[k] = v if v is not None else np.nan
            row["amt_to_card1_mean_ratio"] = (
                amt / card1_stats["card1_amt_mean"]
                if card1_stats["card1_amt_mean"]
                else np.nan
            )
        else:
            # unseen card1 -> all NaN
            for k in [
                "card1_count", "card1_amt_mean", "card1_amt_std",
                "card1_amt_max", "card1_nunique_productcd",
                "amt_to_card1_mean_ratio",
            ]:
                row[k] = np.nan

        # --- 4. UID reconstruction
        d1 = row.get("D1")
        addr1 = row.get("addr1")
        if pd.notna(card1) and pd.notna(addr1) and pd.notna(d1):
            d1n = float(d1) - day
            row["D1n"] = d1n
            uid = f"{int(card1)}_{int(addr1)}_{int(d1n)}"
            row["uid"] = uid
        else:
            row["D1n"] = np.nan
            row["uid"] = None

        # --- 5. UID aggregations from lookup
        uid_stats = self.uid_agg_lookup.get(row["uid"]) if row["uid"] else None
        if uid_stats:
            for k, v in uid_stats.items():
                row[k] = v if v is not None else np.nan
            row["amt_to_uid_mean_ratio"] = (
                amt / uid_stats["uid_amt_mean"]
                if uid_stats["uid_amt_mean"]
                else np.nan
            )
        else:
            for k in [
                "uid_count", "uid_amt_mean", "uid_amt_std",
                "uid_amt_max", "uid_nunique_productcd",
                "amt_to_uid_mean_ratio",
            ]:
                row[k] = np.nan

        # --- 6. Frequency encoding
        for col, lookup in self.freq_lookups.items():
            val = row.get(col)
            key = val if not (pd.isna(val) if val is not None else True) else None
            row[f"{col}_freq"] = lookup.get(key, np.nan)

        # --- 7. Target encoding
        for col, lookup in self.te_lookups.items():
            val = row.get(col)
            key = val if not (pd.isna(val) if val is not None else True) else None
            row[f"{col}_te"] = lookup.get(key, self.global_target_mean)

        # --- 8. Build DataFrame with selected_features in correct order
        out = pd.DataFrame([row])[self.selected_features].copy()

        # --- 9. Restore dtypes
        # 9a. Categorical dtypes with exactly the same category universe
        for col in self.cat_cols:
            out[col] = pd.Categorical(
                out[col], categories=self.cat_categories[col]
            )
        # 9b. Numeric dtypes (JSON None -> Python None -> pandas object,
        # needs explicit cast to float for LGBM)
        for col, dtype_str in self.numeric_dtypes.items():
            if col in out.columns:
                # Use float32 fallback if original was an integer type but
                # NaN sneaked in (int can't hold NaN, float can)
                try:
                    out[col] = out[col].astype(dtype_str)
                except (ValueError, TypeError):
                    out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

        return out

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "FraudPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj