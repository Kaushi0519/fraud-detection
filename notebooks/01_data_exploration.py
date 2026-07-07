"""
Phase 1: Data exploration for the Kaggle "Credit Card Transactions Fraud
Detection Dataset" (kartik2112/fraud-detection).

Answers the basic questions we need before touching feature engineering:
shape, class balance, missing values, and a few key distributions.
"""

import pandas as pd

TRAIN_PATH = "data/raw/fraudTrain.csv"
TEST_PATH = "data/raw/fraudTest.csv"


def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col=0)
    test = pd.read_csv(TEST_PATH, index_col=0)
    return train, test


def summarize(df: pd.DataFrame, name: str):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nColumns and dtypes:")
    print(df.dtypes)

    print("\nClass balance (is_fraud):")
    counts = df["is_fraud"].value_counts()
    pct = df["is_fraud"].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"  {label}: {counts[label]:,} ({pct[label]:.3f}%)")

    print("\nMissing values (columns with any nulls):")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing if not missing.empty else "  None")

    print("\nTransaction amount (amt) distribution:")
    print(df["amt"].describe())


if __name__ == "__main__":
    train_df, test_df = load_data()
    summarize(train_df, "TRAIN")
    summarize(test_df, "TEST")
