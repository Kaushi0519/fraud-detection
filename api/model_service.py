"""
Loads the trained model + frozen feature params once at process startup,
and scores individual transactions.

Reuses features.engineering for the parts of feature computation that are
identical between training and serving (time features, distance, category
rate lookup). amt_vs_user_avg and time_since_last_trans_sec can't reuse the
training implementations directly - those use groupby/expanding over a
whole batch dataframe, which has no meaning for a single incoming request.
Instead they look the user up in a frozen snapshot (see
models/build_serving_artifacts.py); this is the seam Phase 6 replaces with
a live Redis-cached feature store.
"""

import json

import pandas as pd
import shap
import xgboost as xgb

from api.schemas import FeatureContribution, PredictionResponse, TransactionRequest
from features.engineering import (
    FEATURE_COLUMNS,
    add_distance_from_home,
    add_time_features,
    apply_category_fraud_rate,
)

MODEL_PATH = "models/xgboost_fraud_model.json"
ARTIFACTS_PATH = "models/serving_artifacts.json"
FRAUD_THRESHOLD = 0.5
TOP_N_FEATURES = 3


class ModelService:
    def __init__(self, model_path: str = MODEL_PATH, artifacts_path: str = ARTIFACTS_PATH):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)

        with open(artifacts_path) as f:
            artifacts = json.load(f)
        self.category_fraud_rates = artifacts["category_fraud_rates"]
        self.time_gap_fill_sec = artifacts["time_gap_fill_sec"]
        self.default_amt = artifacts["default_amt"]
        self.user_snapshot = artifacts["user_snapshot"]

    def _user_stats(self, cc_num: int) -> dict:
        return self.user_snapshot.get(str(cc_num))

    def _build_features(self, txn: TransactionRequest) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                {
                    "amt": txn.amt,
                    "category": txn.category,
                    "lat": txn.lat,
                    "long": txn.long,
                    "merch_lat": txn.merch_lat,
                    "merch_long": txn.merch_long,
                    "trans_date_trans_time": txn.trans_date_trans_time,
                }
            ]
        )

        df = add_time_features(df)
        df = add_distance_from_home(df)
        df = apply_category_fraud_rate(df, self.category_fraud_rates)

        user_stats = self._user_stats(txn.cc_num)
        if user_stats is None:
            # Unseen user: no history to compare against, so fall back to
            # the same neutral defaults training used for a user's first
            # transaction (see features/engineering.py).
            user_avg_amt = self.default_amt
            time_since_last_sec = self.time_gap_fill_sec
        else:
            user_avg_amt = user_stats["avg_amt"]
            last_trans_time = pd.Timestamp(user_stats["last_trans_time"])
            time_since_last_sec = (
                txn.trans_date_trans_time - last_trans_time
            ).total_seconds()

        df["amt_vs_user_avg"] = df["amt"] / user_avg_amt
        df["time_since_last_trans_sec"] = time_since_last_sec

        return df[FEATURE_COLUMNS]

    def predict(self, txn: TransactionRequest) -> PredictionResponse:
        features = self._build_features(txn)

        fraud_score = float(self.model.predict_proba(features)[0, 1])
        shap_values = self.explainer.shap_values(features)[0]

        contributions = pd.Series(shap_values, index=features.columns)
        top = contributions.abs().sort_values(ascending=False).head(TOP_N_FEATURES)
        top_features = [
            FeatureContribution(
                feature=feat,
                value=float(features.iloc[0][feat]),
                shap_contribution=float(contributions[feat]),
            )
            for feat in top.index
        ]

        return PredictionResponse(
            fraud_score=fraud_score,
            is_fraud=fraud_score >= FRAUD_THRESHOLD,
            top_features=top_features,
        )
