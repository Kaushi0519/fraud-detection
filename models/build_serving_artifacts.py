"""
Builds the artifacts the API needs to score a transaction that arrives with
no history attached: the frozen fit_feature_params() from training (category
fraud rates, time-gap fill value), plus a snapshot of each known user's
latest running stats (average amount so far, last transaction time) as of
the end of the training set.

This user snapshot is a stand-in for a real feature store. Phase 6 replaces
it with Postgres (durable transaction history) + Redis (cached per-user
stats, updated after every scored transaction) - this file exists so the
API has something correct to score against before that infrastructure
exists.
"""

import json

from features.engineering import fit_feature_params, load_raw

TRAIN_PATH = "data/raw/fraudTrain.csv"
OUT_PATH = "models/serving_artifacts.json"


def build_user_snapshot(train_df) -> dict:
    # count is kept alongside avg_amt so the API can do a proper
    # count-weighted blend of this training-time baseline with live
    # Postgres history, instead of one abruptly replacing the other the
    # moment a user's first live transaction arrives.
    latest = train_df.sort_values("trans_date_trans_time").groupby("cc_num").agg(
        avg_amt=("amt", "mean"),
        count=("amt", "size"),
        last_trans_time=("trans_date_trans_time", "max"),
    )
    return {
        str(cc_num): {
            "avg_amt": row["avg_amt"],
            "count": int(row["count"]),
            "last_trans_time": row["last_trans_time"].isoformat(),
        }
        for cc_num, row in latest.iterrows()
    }


if __name__ == "__main__":
    train_df = load_raw(TRAIN_PATH)
    params = fit_feature_params(train_df)

    artifacts = {
        "category_fraud_rates": params["category_fraud_rates"],
        "time_gap_fill_sec": params["time_gap_fill_sec"],
        "default_amt": float(train_df["amt"].mean()),
        "user_snapshot": build_user_snapshot(train_df),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(artifacts, f)
    print(f"Saved serving artifacts for {len(artifacts['user_snapshot'])} users to {OUT_PATH}")
