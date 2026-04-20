"""Pydantic schemas for the fraud detection API.

The request schema is intentionally lax: a real fraud API needs to accept
whatever raw fields the payment system sends, without forcing the client
to populate every V-column. We validate the 'core' fields strictly and
allow arbitrary extra fields to pass through to the preprocessor.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """A raw transaction as it arrives from the payment processor.

    Only the fields needed by our feature engineering are listed explicitly.
    Anything else (V-columns, id-columns, card2-5, addr2, etc.) can be
    passed as extra fields — they'll be picked up by the preprocessor or
    ignored if not in the selected feature set.
    """
    model_config = ConfigDict(extra="allow")  # accept unknown fields silently

    # --- Required core fields
    TransactionID: int = Field(
        ...,
        description="Unique transaction ID from the payment system.",
        examples=[3663549],
    )
    TransactionDT: float = Field(
        ...,
        ge=0,
        description="Seconds since the reference timestamp of the dataset.",
        examples=[15000000.0],
    )
    TransactionAmt: float = Field(
        ...,
        ge=0,
        description="Transaction amount in USD.",
        examples=[49.99],
    )

    # --- Card / address / identity — used to build UID
    card1: Optional[int] = Field(None, description="Card ID (~BIN level).")
    addr1: Optional[float] = Field(None, description="Primary address region.")
    D1: Optional[float] = Field(
        None,
        description="Days since card registration (used for D1n and UID).",
    )
    ProductCD: Optional[str] = Field(
        None,
        description="Product category code (W, C, H, R, S).",
    )

    # --- Categorical fields commonly used by the model
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceInfo: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None


class PredictionResponse(BaseModel):
    """Fraud prediction for a single transaction."""

    transaction_id: int = Field(
        ...,
        description="Echo of the input TransactionID for request correlation.",
    )
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's probability that the transaction is fraud.",
    )
    label: Literal["fraud", "not_fraud"] = Field(
        ...,
        description="Binary decision derived from fraud_probability via threshold.",
    )
    threshold: float = Field(
        ...,
        description="Threshold used to derive the label.",
    )
    model_version: str = Field(
        ...,
        description="Human-readable model identifier (for audit/debugging).",
    )


class HealthResponse(BaseModel):
    """Service health status."""

    status: Literal["ok", "not_ready"]
    model_loaded: bool
    preprocessor_loaded: bool