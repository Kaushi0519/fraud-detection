from datetime import datetime

import pandas as pd
import pytest

from features.engineering import (
    add_amount_vs_user_avg,
    add_distance_from_home,
    add_time_features,
    add_time_since_last_transaction,
    apply_category_fraud_rate,
    fit_category_fraud_rates,
    fit_time_gap_fill,
)


def test_add_time_features_extracts_hour_and_day_of_week():
    # 2020-06-21 was a Sunday.
    df = pd.DataFrame({"trans_date_trans_time": [pd.Timestamp("2020-06-21 22:06:39")]})
    result = add_time_features(df)
    assert result["trans_hour"].iloc[0] == 22
    assert result["trans_day_of_week"].iloc[0] == 6


def test_distance_from_home_is_zero_for_identical_points():
    df = pd.DataFrame(
        {"lat": [40.0], "long": [-74.0], "merch_lat": [40.0], "merch_long": [-74.0]}
    )
    result = add_distance_from_home(df)
    assert result["distance_from_home_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_distance_from_home_matches_known_distance():
    # NYC to Philadelphia is roughly 130km.
    df = pd.DataFrame(
        {"lat": [40.7128], "long": [-74.0060], "merch_lat": [39.9526], "merch_long": [-75.1652]}
    )
    result = add_distance_from_home(df)
    assert result["distance_from_home_km"].iloc[0] == pytest.approx(130, rel=0.1)


def test_amount_vs_user_avg_first_transaction_gets_neutral_ratio():
    df = pd.DataFrame(
        {
            "cc_num": [1],
            "amt": [50.0],
            "trans_date_trans_time": [pd.Timestamp("2020-01-01")],
        }
    )
    result = add_amount_vs_user_avg(df)
    # No history yet - fallback is the transaction's own amount, so ratio is 1.0.
    assert result["amt_vs_user_avg"].iloc[0] == pytest.approx(1.0)


def test_amount_vs_user_avg_excludes_current_transaction_from_baseline():
    df = pd.DataFrame(
        {
            "cc_num": [1, 1, 1],
            "amt": [10.0, 10.0, 100.0],
            "trans_date_trans_time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
        }
    )
    result = add_amount_vs_user_avg(df)
    # Third transaction's baseline should be the average of the first two
    # ($10) only - not including itself ($100), which would understate how
    # unusual it is.
    assert result["user_avg_amt_so_far"].iloc[2] == pytest.approx(10.0)
    assert result["amt_vs_user_avg"].iloc[2] == pytest.approx(10.0)


def test_time_since_last_transaction_fills_first_row_with_provided_value():
    df = pd.DataFrame(
        {
            "cc_num": [1],
            "trans_date_trans_time": [pd.Timestamp("2020-01-01")],
        }
    )
    result = add_time_since_last_transaction(df, fill_value=999.0)
    assert result["time_since_last_trans_sec"].iloc[0] == 999.0


def test_time_since_last_transaction_computes_real_gap():
    df = pd.DataFrame(
        {
            "cc_num": [1, 1],
            "trans_date_trans_time": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 00:10:00"),
            ],
        }
    )
    result = add_time_since_last_transaction(df, fill_value=0.0)
    assert result["time_since_last_trans_sec"].iloc[1] == 600.0


def test_fit_category_fraud_rates_and_apply_with_unseen_category_fallback():
    train_df = pd.DataFrame(
        {"category": ["grocery", "grocery", "shopping"], "is_fraud": [0, 1, 0]}
    )
    rates = fit_category_fraud_rates(train_df)
    assert rates["grocery"] == pytest.approx(0.5)
    assert rates["shopping"] == pytest.approx(0.0)

    test_df = pd.DataFrame({"category": ["grocery", "never_seen_category"]})
    result = apply_category_fraud_rate(test_df, rates)
    assert result["category_fraud_rate"].iloc[0] == pytest.approx(0.5)
    # Unseen category falls back to the overall training fraud rate, not NaN.
    assert result["category_fraud_rate"].iloc[1] == pytest.approx(rates["__default__"])


def test_fit_time_gap_fill_ignores_first_transaction_per_user():
    train_df = pd.DataFrame(
        {
            "cc_num": [1, 1, 2],
            "trans_date_trans_time": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 00:20:00"),
                pd.Timestamp("2020-01-01 00:00:00"),
            ],
        }
    )
    # Only user 1's second transaction has a real gap (1200s); user 2's
    # single transaction and user 1's first contribute no gap.
    assert fit_time_gap_fill(train_df) == pytest.approx(1200.0)
