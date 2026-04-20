"""Singleton model registry — loads model and preprocessor once at startup.

FastAPI calls functions on every request, so loading the model inside
the endpoint would slow everything down. Instead, we load it once in
a module-level variable and expose getter functions.

Tests can call `set_artifacts()` to inject mocks without touching disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightgbm as lgb

from src.inference.preprocessor import FraudPreprocessor


# Module-level state
_model: Optional[lgb.Booster] = None
_preprocessor: Optional[FraudPreprocessor] = None


def load_artifacts(
    model_path: str | Path = "models/lgbm_v5.txt",
    preprocessor_path: str | Path = "models/preprocessor.pkl",
) -> None:
    """Load model and preprocessor from disk. Called once at service startup."""
    global _model, _preprocessor
    _model = lgb.Booster(model_file=str(model_path))
    _preprocessor = FraudPreprocessor.load(preprocessor_path)


def set_artifacts(
    model: lgb.Booster,
    preprocessor: FraudPreprocessor,
) -> None:
    """Inject artifacts directly (used by tests to bypass disk loading)."""
    global _model, _preprocessor
    _model = model
    _preprocessor = preprocessor


def get_model() -> lgb.Booster:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")
    return _model


def get_preprocessor() -> FraudPreprocessor:
    if _preprocessor is None:
        raise RuntimeError("Preprocessor not loaded. Call load_artifacts() first.")
    return _preprocessor


def is_ready() -> bool:
    """Check both artifacts are loaded (for /health endpoint)."""
    return _model is not None and _preprocessor is not None