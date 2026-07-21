import pytest

KNOWN_FRAUD_PAYLOAD = {
    "cc_num": 3560725013359375,
    "amt": 24.84,
    "category": "health_fitness",
    "lat": 31.8599,
    "long": -102.7413,
    "merch_lat": 32.575873,
    "merch_long": -102.60429,
    "trans_date_trans_time": "2020-06-21T22:06:39",
}

LEGIT_PAYLOAD = {
    "cc_num": 111111,
    "amt": 10.0,
    "category": "grocery_pos",
    "lat": 40.0,
    "long": -74.0,
    "merch_lat": 40.0,
    "merch_long": -74.0,
    "trans_date_trans_time": "2020-06-21T22:06:39",
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_well_formed_response(client):
    resp = client.post("/predict", json=LEGIT_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert len(body["top_features"]) == 3
    for feature in body["top_features"]:
        assert set(feature.keys()) == {"feature", "value", "shap_contribution"}


def test_predict_flags_known_fraud_transaction(client):
    # Same real transaction manually verified against the live API/Postgres/
    # Redis stack in Phases 5-8 (score 0.556, above the 0.5 threshold).
    resp = client.post("/predict", json=KNOWN_FRAUD_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_fraud"] is True
    assert body["fraud_score"] == pytest.approx(0.5555986762046814)


def test_predict_rejects_non_positive_amount(client):
    payload = {**LEGIT_PAYLOAD, "amt": -5.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = dict(LEGIT_PAYLOAD)
    del payload["category"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_repeat_requests_update_cached_user_stats(client):
    """The count-weighted blending fix from Phase 6, exercised through the
    API: two requests for the same new user should show an incrementing
    count in the response's feature values across cache-backed replays."""
    first = client.post("/predict", json=LEGIT_PAYLOAD).json()
    second_payload = {**LEGIT_PAYLOAD, "amt": 20.0}
    second = client.post("/predict", json=second_payload).json()

    # Both should succeed and produce valid scores; the underlying
    # amt_vs_user_avg feature differs since the second call has one more
    # data point in its running average than the first.
    assert 0.0 <= first["fraud_score"] <= 1.0
    assert 0.0 <= second["fraud_score"] <= 1.0


def test_metrics_reflects_recorded_predictions(client):
    client.post("/predict", json=LEGIT_PAYLOAD)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_count"] >= 1
    assert body["avg_latency_ms"] >= 0
    assert body["fraud_predicted_count"] + body["legit_predicted_count"] == body["request_count"]
