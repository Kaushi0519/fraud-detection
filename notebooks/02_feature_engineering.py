"""
Phase 2: Feature engineering (draft).

Builds the first 5 fraud-signal features on top of the raw transaction data:
1. amt_vs_user_avg      - how unusual is this amount for this specific user?
2. time_since_last_trans - how soon after their last transaction did this happen?
3. distance_from_home   - how far is the merchant from the customer's home?
4. trans_hour / trans_day_of_week - time-of-day / day-of-week patterns
5. category_fraud_rate  - how risky is this merchant category historically?
"""

import numpy as np
import pandas as pd

TRAIN_PATH = "data/raw/fraudTrain.csv"


def load_data():
    df = pd.read_csv(TRAIN_PATH, index_col=0)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    # Sort by user then time - several features below depend on transaction order.
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # Mon=0..Sun=6
    return df


def add_amount_vs_user_avg(df: pd.DataFrame) -> pd.DataFrame:
    # groupby("cc_num")["amt"] groups all transactions by card/user, then we
    # need each row's "average of THIS USER's transactions BEFORE this one".
    # Using expanding().mean() gives a running average as of each row, and
    # .shift(1) excludes the current transaction itself - otherwise a fraud
    # transaction would be included in its own "normal" baseline (leakage).
    running_avg = (
        df.groupby("cc_num")["amt"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    df["user_avg_amt_so_far"] = running_avg
    # First transaction for a user has no history yet -> NaN. Fill with the
    # transaction's own amount (i.e. "no deviation") rather than dropping rows.
    df["user_avg_amt_so_far"] = df["user_avg_amt_so_far"].fillna(df["amt"])
    df["amt_vs_user_avg"] = df["amt"] / df["user_avg_amt_so_far"]
    return df


def add_time_since_last_transaction(df: pd.DataFrame) -> pd.DataFrame:
    # Within each user's transactions (already sorted by time), diff() gives
    # the gap to the previous row. First transaction per user has no previous
    # one, so it's NaN - fill with a large value (no recent activity).
    df["time_since_last_trans_sec"] = (
        df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    )
    df["time_since_last_trans_sec"] = df["time_since_last_trans_sec"].fillna(
        df["time_since_last_trans_sec"].median()
    )
    return df


def add_distance_from_home(df: pd.DataFrame) -> pd.DataFrame:
    # Haversine formula: great-circle distance between two lat/long points,
    # accounting for the Earth's curvature (straight-line lat/long difference
    # would be wrong since degrees of longitude shrink near the poles).
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
    """Compute each category's historical fraud rate from TRAINING data only.

    This mirrors sklearn's fit/transform split: the rates are learned once
    from train_df, then frozen and reused everywhere else (test set, live
    scoring in Phase 5/6) via apply_category_fraud_rate(). Recomputing this
    on data that includes the row being scored would leak the row's own
    label into its own feature.
    """
    rates = train_df.groupby("category")["is_fraud"].mean().to_dict()
    # Fallback for a category seen in test/production but never in train.
    rates["__default__"] = train_df["is_fraud"].mean()
    return rates


def apply_category_fraud_rate(df: pd.DataFrame, category_rates: dict) -> pd.DataFrame:
    default = category_rates["__default__"]
    df["category_fraud_rate"] = df["category"].map(category_rates).fillna(default)
    return df


def build_features(df: pd.DataFrame, category_rates: dict) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_amount_vs_user_avg(df)
    df = add_time_since_last_transaction(df)
    df = add_distance_from_home(df)
    df = apply_category_fraud_rate(df, category_rates)
    return df


if __name__ == "__main__":
    df = load_data()
    # Fit on the same data we're transforming here since this is all "train"
    # for now. Once we have a real train/test split (Phase 3), category_rates
    # must be fit on the train split only, then applied to both.
    category_rates = fit_category_fraud_rates(df)
    df = build_features(df, category_rates)
    cols = [
        "cc_num",
        "trans_date_trans_time",
        "amt",
        "amt_vs_user_avg",
        "time_since_last_trans_sec",
        "distance_from_home_km",
        "trans_hour",
        "trans_day_of_week",
        "category_fraud_rate",
        "is_fraud",
    ]
    print(df[cols].head(15))
    print("\nNull counts in new features:")
    print(df[cols].isnull().sum())
