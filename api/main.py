"""
Phase 5: FastAPI scoring endpoint.

The model service loads the trained XGBoost model + frozen feature params
once at startup (not per-request) - both are static until the next
retrain, so reloading them on every call would be pure wasted latency.
"""

from fastapi import FastAPI

from api.model_service import ModelService
from api.schemas import PredictionResponse, TransactionRequest

app = FastAPI(title="Fraud Detection API")
model_service = ModelService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    return model_service.predict(transaction)
