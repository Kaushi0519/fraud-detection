"""
Phase 3: Baseline logistic regression model.

Uses Kaggle's native fraudTrain/fraudTest split (rather than re-splitting
fraudTrain ourselves) so it doubles as the real train/test boundary the
feature engineering fit/transform split was designed for.

Evaluated on precision, recall, F1, ROC-AUC, and a confusion matrix -
NOT accuracy, which is meaningless here: with ~99.4% of transactions
legitimate, a model that always predicts "not fraud" scores ~99.4%
accuracy while catching zero fraud.
"""

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.engineering import FEATURE_COLUMNS, fit_feature_params, load_raw, transform

TRAIN_PATH = "data/raw/fraudTrain.csv"
TEST_PATH = "data/raw/fraudTest.csv"
METRICS_OUT_PATH = "models/baseline_metrics.json"


def prepare_data():
    train_df = load_raw(TRAIN_PATH)
    test_df = load_raw(TEST_PATH)

    params = fit_feature_params(train_df)
    train_df = transform(train_df, params)
    test_df = transform(test_df, params)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_fraud"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_fraud"]
    return X_train, y_train, X_test, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    # class_weight="balanced" reweights the loss inversely proportional to
    # class frequency, so the ~0.58% fraud class isn't drowned out. This is
    # the simplest imbalance strategy - SMOTE is evaluated against it in
    # Phase 4 for the production model.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{metrics['confusion_matrix']}")

    return metrics


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_OUT_PATH}")
