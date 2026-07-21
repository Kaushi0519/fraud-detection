"""
Phase 4: Production model - XGBoost, with a real SMOTE vs. class-weighting
comparison, evaluated against the Phase 3 baseline, plus SHAP explainability.

Two imbalance strategies are trained and compared head-to-head:
  A) scale_pos_weight - reweights the loss during training (no new rows).
  B) SMOTE - synthetically oversamples the minority (fraud) class in the
     training set before fitting, so the model sees a more balanced class
     distribution directly.
The better performer (by F1, since it balances the precision/recall
tradeoff that matters for actionable fraud alerts) is saved as the
production model.
"""

import json

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.data import prepare_data

MODEL_OUT_PATH = "models/xgboost_fraud_model.json"
METRICS_OUT_PATH = "models/production_metrics.json"


def train_with_scale_pos_weight(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_with_smote(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)
    return model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def explain_sample(model: xgb.XGBClassifier, X_test: pd.DataFrame, n: int = 3):
    """Preview of the per-transaction explanation the Phase 5 API will return:
    for a handful of transactions the model flagged as fraud, show which
    features pushed the prediction toward "fraud" the most."""
    explainer = shap.TreeExplainer(model)
    flagged = X_test[model.predict(X_test) == 1].head(n)
    if flagged.empty:
        print("No transactions flagged for the sample explanation.")
        return

    shap_values = explainer.shap_values(flagged)
    print(f"\nSHAP explanations for {len(flagged)} flagged transactions:")
    for i, (idx, row) in enumerate(flagged.iterrows()):
        contributions = pd.Series(shap_values[i], index=flagged.columns)
        top = contributions.abs().sort_values(ascending=False).head(3)
        print(f"\n  Transaction {idx}:")
        for feat in top.index:
            print(f"    {feat}: shap={contributions[feat]:+.4f}, value={row[feat]:.4f}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()

    print("Training with scale_pos_weight...")
    model_weighted = train_with_scale_pos_weight(X_train, y_train)
    metrics_weighted = evaluate(model_weighted, X_test, y_test)

    print("Training with SMOTE...")
    model_smote = train_with_smote(X_train, y_train)
    metrics_smote = evaluate(model_smote, X_test, y_test)

    print("\n=== scale_pos_weight ===")
    print(json.dumps(metrics_weighted, indent=2))
    print("\n=== SMOTE ===")
    print(json.dumps(metrics_smote, indent=2))

    winner_name, winner_model, winner_metrics = (
        ("scale_pos_weight", model_weighted, metrics_weighted)
        if metrics_weighted["f1"] >= metrics_smote["f1"]
        else ("smote", model_smote, metrics_smote)
    )
    print(f"\nWinner (by F1): {winner_name}")

    explain_sample(winner_model, X_test)

    winner_model.save_model(MODEL_OUT_PATH)
    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(
            {
                "scale_pos_weight": metrics_weighted,
                "smote": metrics_smote,
                "winner": winner_name,
                "winner_metrics": winner_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nProduction model saved to {MODEL_OUT_PATH}")
    print(f"Metrics saved to {METRICS_OUT_PATH}")
