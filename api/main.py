"""
Phase 5/6: FastAPI scoring endpoint, backed by PostgreSQL (transaction
history) and Redis (cached per-user features).

The model service loads the trained XGBoost model + frozen feature params
once at startup (not per-request) - both are static until the next
retrain, so reloading them on every call would be pure wasted latency.
"""

from typing import Generator

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from api import db
from api.model_service import ModelService
from api.schemas import PredictionResponse, TransactionRequest

app = FastAPI(title="Fraud Detection API")
model_service = ModelService()


@app.on_event("startup")
def on_startup():
    db.init_db()


def get_db() -> Generator[Session, None, None]:
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, session: Session = Depends(get_db)):
    return model_service.predict(transaction, session)
