"""Unit tests for FraudPreprocessor.

These tests exercise the transform() contract and a few fit()-level
invariants. They do NOT check prediction quality — that's measured on
Kaggle LB, not in CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.preprocessor import FraudPreprocessor


# -------------------------------------------------------------------- fit()

def test_fit_populates_all_lookups(fitted_preprocessor):
    """After fit(), every lookup table and column list must be non-empty."""
    pre = fitted_preprocessor
    assert pre._fitted is True
    assert len(pre.selected_features) > 0
    assert len(pre.freq_lookups) > 0
    assert len(pre.te_lookups) > 0
    assert len(pre.uid_agg_lookup) > 0
    assert len(pre.card1_agg_lookup) > 0
    assert 0.0 <= pre.global_target_mean <= 1.0


def test_fit_separates_numeric_and_categorical(fitted_preprocessor):
    """Every selected feature must be classified as numeric OR categorical,
    never both, never neither."""
    pre = fitted_preprocessor
    num_set = set(pre.numeric_dtypes.keys())
    cat_set = set(pre.cat_cols)
    all_set = set(pre.selected_features)

    assert num_set & cat_set == set(), "feature classified as both numeric and categorical"
    assert num_set | cat_set == all_set, "some features classified as neither"


# -------------------------------------------------------- transform() shape

def test_transform_returns_single_row(fitted_preprocessor, sample_transaction):
    """transform() must return a DataFrame of shape (1, n_selected_features)."""
    out = fitted_preprocessor.transform(sample_transaction)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, len(fitted_preprocessor.selected_features))


def test_transform_preserves_feature_order(fitted_preprocessor, sample_transaction):
    """Column order must match selected_features exactly — the model
    relies on positional feature ordering."""
    out = fitted_preprocessor.transform(sample_transaction)
    assert list(out.columns) == fitted_preprocessor.selected_features


# -------------------------------------------------------- transform() dtypes

def test_transform_numeric_columns_are_numeric(fitted_preprocessor, sample_transaction):
    """Every non-categorical column must end up numeric
    (not object, not string)."""
    out = fitted_preprocessor.transform(sample_transaction)
    for col in fitted_preprocessor.numeric_dtypes.keys():
        assert pd.api.types.is_numeric_dtype(out[col]), \
            f"{col} is {out[col].dtype}, expected numeric"


def test_transform_categorical_columns_match_train_categories(
    fitted_preprocessor, sample_transaction
):
    """Every categorical column must use the same category universe
    that was seen during training (LightGBM requires this)."""
    out = fitted_preprocessor.transform(sample_transaction)
    for col in fitted_preprocessor.cat_cols:
        assert isinstance(out[col].dtype, pd.CategoricalDtype), \
            f"{col} is {out[col].dtype}, expected categorical"
        assert list(out[col].cat.categories) == fitted_preprocessor.cat_categories[col]


# -------------------------------------------------------- transform() robustness

def test_transform_handles_unseen_card1(fitted_preprocessor, sample_transaction):
    """An unseen card1 value must not crash; card1_* features should be NaN."""
    raw = dict(sample_transaction)
    raw["card1"] = 99999  # never seen in synthetic train
    out = fitted_preprocessor.transform(raw)
    # Card1 aggs should be NaN because the lookup misses
    assert pd.isna(out["card1_amt_mean"].iloc[0])


def test_transform_handles_missing_addr1(fitted_preprocessor, sample_transaction):
    """Missing addr1 -> UID is None -> uid_* features must be NaN, no crash."""
    raw = dict(sample_transaction)
    raw["addr1"] = None
    out = fitted_preprocessor.transform(raw)
    assert pd.isna(out["uid_count"].iloc[0])


def test_transform_handles_unseen_category(fitted_preprocessor, sample_transaction):
    """An unseen value in a categorical column must become NaN in the
    output category, not raise an error."""
    raw = dict(sample_transaction)
    raw["P_emaildomain"] = "totally-new-domain.xyz"
    out = fitted_preprocessor.transform(raw)
    # TE-encoded column uses global_mean fallback, so not NaN
    # But the raw categorical column itself should be NaN
    # (unseen value not in cat_categories[col])
    if "P_emaildomain" in fitted_preprocessor.cat_cols:
        assert pd.isna(out["P_emaildomain"].iloc[0])


# -------------------------------------------------------------- save / load

def test_save_and_load_round_trip(fitted_preprocessor, sample_transaction, tmp_path):
    """Pickled preprocessor after load must produce identical output."""
    path = tmp_path / "preprocessor.pkl"
    fitted_preprocessor.save(path)

    reloaded = FraudPreprocessor.load(path)
    original = fitted_preprocessor.transform(sample_transaction)
    restored = reloaded.transform(sample_transaction)

    # Compare numerically — DataFrames with NaN need special handling
    pd.testing.assert_frame_equal(original, restored)


def test_load_wrong_file_raises(tmp_path):
    """Loading a non-preprocessor pickle must raise, not silently succeed."""
    import pickle
    wrong_path = tmp_path / "wrong.pkl"
    with open(wrong_path, "wb") as f:
        pickle.dump({"not": "a preprocessor"}, f)

    with pytest.raises(TypeError):
        FraudPreprocessor.load(wrong_path)


# ----------------------------------------------------------- guard-rails

def test_transform_before_fit_raises():
    """Calling transform() on an unfit preprocessor must raise RuntimeError."""
    pre = FraudPreprocessor()
    with pytest.raises(RuntimeError):
        pre.transform({"TransactionID": 1, "TransactionDT": 0, "TransactionAmt": 1.0})
