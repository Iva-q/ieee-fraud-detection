"""Integration tests for the FastAPI fraud detection service.

Uses FastAPI's TestClient which calls endpoints directly in-process,
no real HTTP server needed. The model registry is pre-populated via
the `api_client` fixture in conftest.py, so we do NOT exercise the
real artifact loading from disk here — that's covered by preprocessor
save/load tests separately.
"""

from __future__ import annotations


# ------------------------------------------------------------------ /health

def test_health_returns_ok_when_ready(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["preprocessor_loaded"] is True


# ------------------------------------------------------------ /predict happy path

def test_predict_returns_200_on_valid_transaction(api_client, sample_transaction):
    r = api_client.post("/predict", json=sample_transaction)
    assert r.status_code == 200, r.text
    body = r.json()

    # Contract: all PredictionResponse fields must be present
    expected_keys = {
        "transaction_id",
        "fraud_probability",
        "label",
        "threshold",
        "model_version",
    }
    assert set(body.keys()) == expected_keys


def test_predict_echoes_transaction_id(api_client, sample_transaction):
    r = api_client.post("/predict", json=sample_transaction)
    body = r.json()
    assert body["transaction_id"] == sample_transaction["TransactionID"]


def test_predict_returns_valid_probability(api_client, sample_transaction):
    r = api_client.post("/predict", json=sample_transaction)
    body = r.json()
    p = body["fraud_probability"]
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_label_matches_threshold(api_client, sample_transaction):
    r = api_client.post("/predict", json=sample_transaction)
    body = r.json()
    threshold = body["threshold"]
    prob = body["fraud_probability"]
    if prob >= threshold:
        assert body["label"] == "fraud"
    else:
        assert body["label"] == "not_fraud"


def test_predict_reports_model_version(api_client, sample_transaction):
    r = api_client.post("/predict", json=sample_transaction)
    body = r.json()
    # Model version must be a non-empty string (useful for audit/debugging)
    assert isinstance(body["model_version"], str)
    assert len(body["model_version"]) > 0


# ----------------------------------------------------------- /predict edge cases

def test_predict_accepts_extra_fields(api_client, sample_transaction):
    """The Transaction schema uses extra='allow', so unknown fields
    (e.g. V-columns, id-columns) must not cause a 422."""
    raw = dict(sample_transaction)
    raw["V420"] = 0.123
    raw["some_new_field_we_never_saw"] = "hello"

    r = api_client.post("/predict", json=raw)
    assert r.status_code == 200, r.text


def test_predict_tolerates_missing_optional_fields(api_client, sample_transaction):
    """Fields like DeviceInfo are Optional — their absence must not fail."""
    raw = dict(sample_transaction)
    raw.pop("DeviceInfo", None)
    raw.pop("P_emaildomain", None)
    raw["DeviceInfo"] = None
    raw["P_emaildomain"] = None

    r = api_client.post("/predict", json=raw)
    assert r.status_code == 200, r.text


# ------------------------------------------------------ /predict bad requests

def test_predict_rejects_missing_required_field(api_client, sample_transaction):
    """Removing TransactionAmt must give 422 (Unprocessable Entity)."""
    raw = dict(sample_transaction)
    raw.pop("TransactionAmt")

    r = api_client.post("/predict", json=raw)
    assert r.status_code == 422


def test_predict_rejects_negative_amount(api_client, sample_transaction):
    """TransactionAmt has ge=0 constraint in the schema — negative values
    must be rejected with 422."""
    raw = dict(sample_transaction)
    raw["TransactionAmt"] = -50.0

    r = api_client.post("/predict", json=raw)
    assert r.status_code == 422


def test_predict_rejects_wrong_type(api_client, sample_transaction):
    """TransactionAmt must be numeric; a string should give 422."""
    raw = dict(sample_transaction)
    raw["TransactionAmt"] = "not_a_number"

    r = api_client.post("/predict", json=raw)
    assert r.status_code == 422


def test_predict_rejects_empty_body(api_client):
    r = api_client.post("/predict", json={})
    assert r.status_code == 422
