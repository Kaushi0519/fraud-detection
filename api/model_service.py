"""
Loads the trained model + frozen feature params once at process startup,
and scores individual transactions.

Reuses features.engineering for the parts of feature computation that are
identical between training and serving (time features, distance, category
rate lookup). amt_vs_user_avg and time_since_last_trans_sec can't reuse the
training implementations directly - those use groupby/expanding over a
whole batch dataframe, which has no meaning for a single incoming request.

Per-user history lookup: Redis cache first; on a miss, count-weighted blend
of the frozen training snapshot (models/serving_artifacts.json) with
whatever this system has scored for the user in Postgres so far. Blending
matters: naively letting live Postgres data override the snapshot outright
would mean a user's very first live transaction resets their entire
multi-year training average to just that one data point - a discontinuity
right when the feature is supposed to matter most. After scoring, the
transaction is persisted to Postgres (durable log) and the blended stats
are updated incrementally and written back to the cache, so the next
request for that user is both fast and count-correct.
"""

import json

import pandas as pd
import shap
import xgboost as xgb
from sqlalchemy.orm import Session

from api import cache, db
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

    def _lookup_user_stats(self, cc_num: int, session: Session) -> dict | None:
        """Returns {"avg_amt", "count", "last_trans_time"} or None for a
        true cold start (never seen in the cache, Postgres, or training)."""
        cached = cache.get_user_features(cc_num)
        if cached is not None:
            return cached

        snapshot = self.user_snapshot.get(str(cc_num))
        live = db.get_live_aggregate(session, cc_num)
        if snapshot is None and live is None:
            return None

        snapshot_count = snapshot["count"] if snapshot else 0
        snapshot_sum = snapshot["avg_amt"] * snapshot_count if snapshot else 0.0
        live_count = live["count"] if live else 0
        live_sum = live["sum_amt"] if live else 0.0

        total_count = snapshot_count + live_count
        blended = {
            "avg_amt": (snapshot_sum + live_sum) / total_count,
            "count": total_count,
            "last_trans_time": live["last_trans_time"] if live else snapshot["last_trans_time"],
        }
        cache.set_user_features(
            cc_num, blended["avg_amt"], blended["count"], blended["last_trans_time"]
        )
        return blended

    def _build_features(self, txn: TransactionRequest, user_stats: dict | None) -> pd.DataFrame:
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

        if user_stats is None:
            # True cold start: never seen this user anywhere (cache, DB, or
            # training snapshot). Same neutral defaults training used for a
            # user's first transaction.
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

    def predict(self, txn: TransactionRequest, session: Session) -> PredictionResponse:
        user_stats = self._lookup_user_stats(txn.cc_num, session)
        features = self._build_features(txn, user_stats)

        fraud_score = float(self.model.predict_proba(features)[0, 1])
        is_fraud = fraud_score >= FRAUD_THRESHOLD
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

        db.save_transaction(
            session,
            cc_num=txn.cc_num,
            amt=txn.amt,
            category=txn.category,
            trans_date_trans_time=txn.trans_date_trans_time,
            fraud_score=fraud_score,
            is_fraud=is_fraud,
        )

        # Incremental (count-weighted) update - cheaper and just as correct
        # as re-querying Postgres for a fresh aggregate after every insert.
        old_count = user_stats["count"] if user_stats else 0
        old_avg = user_stats["avg_amt"] if user_stats else 0.0
        new_count = old_count + 1
        new_avg = (old_avg * old_count + txn.amt) / new_count
        cache.set_user_features(
            txn.cc_num, new_avg, new_count, txn.trans_date_trans_time.isoformat()
        )

        return PredictionResponse(
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            top_features=top_features,
        )
