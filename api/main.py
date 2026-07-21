"""
Phase 5/6/9: FastAPI scoring endpoint, backed by PostgreSQL (transaction
history) and Redis (cached per-user features), with basic request metrics.

The model service loads the trained XGBoost model + frozen feature params
once at startup (not per-request) - both are static until the next
retrain, so reloading them on every call would be pure wasted latency.
"""

import time
from contextlib import asynccontextmanager
from typing import Generator

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from api import db
from api.metrics import metrics_tracker
from api.model_service import ModelService
from api.schemas import PredictionResponse, TransactionRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)
model_service = ModelService()


def get_db() -> Generator[Session, None, None]:
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return metrics_tracker.summary()


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, session: Session = Depends(get_db)):
    start = time.perf_counter()
    result = model_service.predict(transaction, session)
    latency_ms = (time.perf_counter() - start) * 1000
    metrics_tracker.record(latency_ms, result.is_fraud)
    return result
