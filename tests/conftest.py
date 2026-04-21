"""Shared pytest fixtures for all test modules.

Provides:
- `sample_raw_train` / `sample_raw_test` / `sample_target` / `sample_folds`:
  synthetic data of ~50 rows with the minimum columns needed by the
  feature-engineering pipeline.
- `fitted_preprocessor`: a FraudPreprocessor already fit on the synthetic
  data. Shared across tests via scope="session" to avoid re-fitting.
- `api_client`: a FastAPI TestClient with the model registry populated
  from synthetic model + preprocessor.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from src.data.split import make_time_folds
from src.inference.preprocessor import FraudPreprocessor


# Columns the raw schema needs to satisfy build_features()
# (anything not listed here will be absent in synthetic data)
REQUIRED_COLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "D1",
    "D2",
    "D3",
    "D15",
    "C1", "C2", "C5", "C9", "C11", "C13", "C14",
    "V70", "V91", "V258",
    "id_30", "id_31", "id_33",
]


def _make_rows(n: int, dt_start: int, rng: np.random.Generator,
               with_fraud_rate: float = 0.05) -> pd.DataFrame:
    """Generate n synthetic transaction rows.

    String-valued columns are cast to `category` to mimic the real dataset
    (which was saved to parquet with category dtypes after reduce_mem_usage).
    """
    df = pd.DataFrame({
        "TransactionID": np.arange(dt_start, dt_start + n),
        "TransactionDT": np.linspace(dt_start * 86400, (dt_start + n) * 86400, n).astype(np.int64),
        "TransactionAmt": rng.uniform(5.0, 500.0, n).astype(np.float32),
        "ProductCD": rng.choice(["W", "C", "H"], n),
        "card1": rng.integers(1000, 1100, n).astype(np.int16),
        "card2": rng.integers(100, 200, n).astype(np.float32),
        "card3": np.full(n, 150, dtype=np.float32),
        "card4": rng.choice(["visa", "mastercard"], n),
        "card5": rng.integers(100, 200, n).astype(np.float32),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.choice([100.0, 200.0, 300.0], n).astype(np.float32),
        "addr2": np.full(n, 87.0, dtype=np.float32),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "R_emaildomain": rng.choice(["gmail.com", None, None], n),
        "DeviceInfo": rng.choice(["Samsung", "Apple", None], n),
        "D1": rng.integers(0, 50, n).astype(np.float32),
        "D2": rng.integers(0, 50, n).astype(np.float32),
        "D3": rng.integers(0, 30, n).astype(np.float32),
        "D15": rng.integers(0, 30, n).astype(np.float32),
        "C1": rng.integers(0, 20, n).astype(np.float32),
        "C2": rng.integers(0, 20, n).astype(np.float32),
        "C5": rng.integers(0, 10, n).astype(np.float32),
        "C9": rng.integers(0, 10, n).astype(np.float32),
        "C11": rng.integers(0, 10, n).astype(np.float32),
        "C13": rng.integers(0, 200, n).astype(np.float32),
        "C14": rng.integers(0, 20, n).astype(np.float32),
        "V70": rng.uniform(0, 1, n).astype(np.float32),
        "V91": rng.uniform(0, 1, n).astype(np.float32),
        "V258": rng.uniform(0, 1, n).astype(np.float32),
        "id_30": rng.choice(["Windows 10", "iOS 13", None], n),
        "id_31": rng.choice(["chrome", "safari", None], n),
        "id_33": rng.choice(["1920x1080", "2560x1440", None], n),
    })
    # Cast string-valued columns to category (mimics the real parquet schema)
    string_cols = [
        "ProductCD", "card4", "card6",
        "P_emaildomain", "R_emaildomain", "DeviceInfo",
        "id_30", "id_31", "id_33",
    ]
    for col in string_cols:
        df[col] = df[col].astype("category")
    return df


@pytest.fixture(scope="session")
def sample_raw_train() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = _make_rows(n=80, dt_start=1, rng=rng)
    # isFraud is only in train
    df["isFraud"] = (rng.uniform(0, 1, 80) < 0.08).astype(np.int8)
    return df


@pytest.fixture(scope="session")
def sample_raw_test() -> pd.DataFrame:
    rng = np.random.default_rng(43)
    return _make_rows(n=40, dt_start=81, rng=rng)


@pytest.fixture(scope="session")
def sample_target(sample_raw_train: pd.DataFrame) -> pd.Series:
    return sample_raw_train["isFraud"].astype(np.int8)


@pytest.fixture(scope="session")
def sample_folds(sample_raw_train: pd.DataFrame) -> np.ndarray:
    return make_time_folds(sample_raw_train["TransactionDT"], n_splits=5)


@pytest.fixture(scope="session")
def fitted_preprocessor(
    sample_raw_train: pd.DataFrame,
    sample_raw_test: pd.DataFrame,
    sample_target: pd.Series,
    sample_folds: np.ndarray,
) -> FraudPreprocessor:
    """Fit a preprocessor on the synthetic data, return it for tests.

    Uses a reduced selected-features list (just those build_features()
    actually produces with our tiny columns) so the transform doesn't
    KeyError on missing V-columns.
    """
    # Run build_features once to see what columns end up in train_fe
    from src.features.build_features import build_features
    train_fe, _ = build_features(
        train=sample_raw_train.copy(),
        test=sample_raw_test.copy(),
        target=sample_target,
        folds=sample_folds,
        verbose=False,
    )
    # Use all non-service columns as "selected"
    drop_cols = ["TransactionID", "isFraud", "TransactionDT", "uid"]
    selected = [c for c in train_fe.columns if c not in drop_cols]

    pre = FraudPreprocessor()
    pre.fit(
        train=sample_raw_train,
        test=sample_raw_test,
        target=sample_target,
        folds=sample_folds,
        selected_features=selected,
    )
    return pre


@pytest.fixture(scope="session")
def trained_model(
    fitted_preprocessor: FraudPreprocessor,
    sample_raw_train: pd.DataFrame,
    sample_raw_test: pd.DataFrame,
    sample_target: pd.Series,
    sample_folds: np.ndarray,
) -> lgb.Booster:
    """Train a tiny LightGBM model on the synthetic data."""
    from src.features.build_features import build_features

    train_fe, _ = build_features(
        train=sample_raw_train.copy(),
        test=sample_raw_test.copy(),
        target=sample_target,
        folds=sample_folds,
        verbose=False,
    )
    y = sample_target.values
    drop_cols = ["TransactionID", "isFraud", "TransactionDT", "uid"]
    X = train_fe.drop(columns=drop_cols)[fitted_preprocessor.selected_features]
    cat_cols = fitted_preprocessor.cat_cols

    train_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
    model = lgb.train(
        params={
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbose": -1,
        },
        train_set=train_data,
        num_boost_round=10,
    )
    return model


@pytest.fixture(scope="session")
def sample_transaction(sample_raw_test: pd.DataFrame) -> dict:
    """One raw transaction dict — as it would arrive in a POST request.

    Converts numpy scalars to plain Python and NaN to None so the dict
    is JSON-serializable (mirrors what a real HTTP client would send).
    """
    row = sample_raw_test.iloc[0].to_dict()
    cleaned = {}
    for k, v in row.items():
        # NaN -> None for JSON compliance
        if isinstance(v, float) and np.isnan(v):
            cleaned[k] = None
        # numpy scalars -> Python scalars
        elif hasattr(v, "item"):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


@pytest.fixture
def api_client(fitted_preprocessor, trained_model):
    """FastAPI TestClient with the model registry populated in-memory."""
    from fastapi.testclient import TestClient
    from app import model_registry, main

    model_registry.set_artifacts(model=trained_model, preprocessor=fitted_preprocessor)
    # Bypass the lifespan loader by using TestClient without lifespan events
    # (set_artifacts() already populated the registry)
    with TestClient(main.app) as client:
        yield client
