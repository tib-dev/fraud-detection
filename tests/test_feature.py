import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

from fraud_detection.features.custom_features import add_fraud_features


def make_sample_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 1],
        "device_id": ["d1", "d1", "d2", "d1"],
        "ip_address": ["ip1", "ip1", "ip2", "ip1"],
        "signup_time": [
            "2024-01-01 08:00:00",
            "2024-01-01 08:00:00",
            "2024-01-02 09:00:00",
            "2024-01-01 08:00:00",
        ],
        "purchase_time": [
            "2024-01-01 09:00:00",
            "2024-01-01 09:30:00",
            "2024-01-02 10:00:00",
            "2024-01-02 09:00:00",
        ],
    })


def test_output_shape_and_order_preserved():
    df = make_sample_df()
    result = add_fraud_features(df)

    assert len(result) == len(df)
    assert (result.index == df.index).all()


def test_time_since_signup_computation():
    df = make_sample_df()
    result = add_fraud_features(df)

    expected_seconds = (
        pd.to_datetime(df["purchase_time"])
        - pd.to_datetime(df["signup_time"])
    ).dt.total_seconds()

    assert_series_equal(
        result["time_since_signup"],
        expected_seconds,
        check_names=False,
    )


def test_hour_and_day_features():
    df = make_sample_df()
    result = add_fraud_features(df)

    assert "hour_of_day" in result.columns
    assert "day_of_week" in result.columns

    assert result["hour_of_day"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()


def test_user_velocity_features_exist_and_non_negative():
    df = make_sample_df()
    result = add_fraud_features(df)

    for col in ["tx_count_uder_id_1h", "tx_count_uder_id_24h"]:
        assert col in result.columns
        assert (result[col] >= 0).all()


def test_velocity_logic_simple_case():
    df = make_sample_df()
    result = add_fraud_features(df)

    # user_id=1 has multiple tx within 24h
    user_1_rows = result[result["user_id"] == 1]

    assert user_1_rows["tx_count_uder_id_24h"].max() >= 2


def test_global_frequency_features():
    df = make_sample_df()
    result = add_fraud_features(df)

    assert (result["device_id_count"] == [3, 3, 1, 3]).all()
    assert (result["ip_address_count"] == [3, 3, 1, 3]).all()
    assert (result["user_total_transactions"] == [3, 3, 1, 3]).all()


def test_no_nan_values_in_generated_features():
    df = make_sample_df()
    result = add_fraud_features(df)

    engineered_cols = [
        "time_since_signup",
        "hour_of_day",
        "day_of_week",
        "tx_count_uder_id_1h",
        "tx_count_uder_id_24h",
        "device_id_count",
        "ip_address_count",
        "user_total_transactions",
    ]

    assert not result[engineered_cols].isna().any().any()
