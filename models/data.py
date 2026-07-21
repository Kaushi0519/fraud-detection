"""Shared train/test data preparation for model training scripts."""

from features.engineering import FEATURE_COLUMNS, fit_feature_params, load_raw, transform

TRAIN_PATH = "data/raw/fraudTrain.csv"
TEST_PATH = "data/raw/fraudTest.csv"


def prepare_data():
    train_df = load_raw(TRAIN_PATH)
    test_df = load_raw(TEST_PATH)

    params = fit_feature_params(train_df)
    train_df = transform(train_df, params)
    test_df = transform(test_df, params)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_fraud"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_fraud"]
    return X_train, y_train, X_test, y_test
