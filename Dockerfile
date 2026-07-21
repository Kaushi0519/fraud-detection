FROM python:3.13-slim

# libgomp1 is the Linux equivalent of the libomp OpenMP runtime XGBoost
# needs (macOS needs `brew install libomp` for the same reason locally).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY features/ features/
COPY models/xgboost_fraud_model.json models/serving_artifacts.json models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
