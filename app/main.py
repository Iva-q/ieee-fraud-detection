"""FastAPI application for fraud detection.

Exposes two endpoints:
- GET  /health   — liveness + readiness check
- POST /predict  — score a single transaction

Model and preprocessor are loaded once at startup via the lifespan
context manager (FastAPI's replacement for deprecated @on_event hooks).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from app import model_registry
from app.schemas import HealthResponse, PredictionResponse, Transaction


# Decision threshold for converting probability to a binary label.
# 0.5 is a naive default; in production this would be tuned on a
# cost-weighted validation set. Documented in the response for auditability.
FRAUD_THRESHOLD = 0.5
MODEL_VERSION = "lgbm_v5"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts at startup, release at shutdown."""
    model_registry.load_artifacts(
        model_path="models/lgbm_v5.txt",
        preprocessor_path="models/preprocessor.pkl",
    )
    yield
    # Nothing to clean up — Python GC handles it


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description=(
        "Single-transaction fraud scoring service. "
        "Returns the probability that a transaction is fraudulent, "
        "plus a binary label derived from a configurable threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + readiness check",
)
async def health() -> HealthResponse:
    ready = model_registry.is_ready()
    return HealthResponse(
        status="ok" if ready else "not_ready",
        model_loaded=model_registry._model is not None,
        preprocessor_loaded=model_registry._preprocessor is not None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single transaction",
)
async def predict(transaction: Transaction) -> PredictionResponse:
    """Score a single transaction for fraud probability.

    The request body must contain at minimum TransactionID, TransactionDT,
    and TransactionAmt. All other fields are optional — the model and
    preprocessor handle missing values natively.
    """
    if not model_registry.is_ready():
        raise HTTPException(status_code=503, detail="Service is not ready")

    # Dump to plain dict — extras (V-columns etc) come along for free
    # because schemas.Transaction has extra='allow'
    raw = transaction.model_dump()

    try:
        features = model_registry.get_preprocessor().transform(raw)
    except Exception as e:
        # Preprocessor failure usually means bad input data (e.g. unseen
        # categorical that crashes somewhere we didn't anticipate).
        # Expose the error cleanly instead of a 500 with a traceback.
        raise HTTPException(
            status_code=422,
            detail=f"Preprocessor failed: {type(e).__name__}: {e}",
        )

    try:
        proba = float(model_registry.get_model().predict(features)[0])
    except Exception as e:
        # Very rare — model shape mismatch, corrupt model file, etc.
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {type(e).__name__}: {e}",
        )

    # Safety: clip to [0, 1] just in case of numerical edge cases.
    # Pydantic will reject otherwise (ge=0, le=1 on the response schema).
    proba = float(np.clip(proba, 0.0, 1.0))

    return PredictionResponse(
        transaction_id=transaction.TransactionID,
        fraud_probability=proba,
        label="fraud" if proba >= FRAUD_THRESHOLD else "not_fraud",
        threshold=FRAUD_THRESHOLD,
        model_version=MODEL_VERSION,
    )