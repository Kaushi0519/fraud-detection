"""
Reusable feature engineering for the fraud detection pipeline.

Generalized from notebooks/02_feature_engineering.py (Phase 2 draft) into
functions shared by model training (Phase 3/4) and, eventually, the API
scoring path (Phase 5/6).

Fit/transform split: any statistic derived from the training distribution
(category fraud rates, the time-gap fill value) is fit on train data only
and passed explicitly into transform, so the same frozen values are reused
on the test set - avoiding the leakage that computing them fresh on each
dataset would cause.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amt",
    "amt_vs_user_avg",
    "time_since_last_trans_sec",
    "distance_from_home_km",
    "trans_hour",
    "trans_day_of_week",
    "category_fraud_rate",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    return df


def add_amount_vs_user_avg(df: pd.DataFrame) -> pd.DataFrame:
    running_avg = (
        df.groupby("cc_num")["amt"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    df["user_avg_amt_so_far"] = running_avg.fillna(df["amt"])
    df["amt_vs_user_avg"] = df["amt"] / df["user_avg_amt_so_far"]
    return df


def fit_time_gap_fill(train_df: pd.DataFrame) -> float:
    """Median seconds-since-last-transaction across train, ignoring each
    user's first (gap-less) transaction. Frozen and reused as the fill
    value for every dataset's first-per-user rows."""
    gaps = train_df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    return float(gaps.median())


def add_time_since_last_transaction(df: pd.DataFrame, fill_value: float) -> pd.DataFrame:
    gaps = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    df["time_since_last_trans_sec"] = gaps.fillna(fill_value)
    return df


def add_distance_from_home(df: pd.DataFrame) -> pd.DataFrame:
    lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
    lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371
    df["distance_from_home_km"] = earth_radius_km * c
    return df


def fit_category_fraud_rates(train_df: pd.DataFrame) -> dict:
    rates = train_df.groupby("category")["is_fraud"].mean().to_dict()
    rates["__default__"] = train_df["is_fraud"].mean()
    return rates


def apply_category_fraud_rate(df: pd.DataFrame, category_rates: dict) -> pd.DataFrame:
    default = category_rates["__default__"]
    df["category_fraud_rate"] = df["category"].map(category_rates).fillna(default)
    return df


def fit_feature_params(train_df: pd.DataFrame) -> dict:
    """Fit every train-derived statistic once. Reuse the returned dict for
    both the train and test transform() calls."""
    return {
        "category_fraud_rates": fit_category_fraud_rates(train_df),
        "time_gap_fill_sec": fit_time_gap_fill(train_df),
    }


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_amount_vs_user_avg(df)
    df = add_time_since_last_transaction(df, params["time_gap_fill_sec"])
    df = add_distance_from_home(df)
    df = apply_category_fraud_rate(df, params["category_fraud_rates"])
    return df
