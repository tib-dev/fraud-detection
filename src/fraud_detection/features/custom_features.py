import pandas as pd


def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_index = df.index

    # -----------------------
    # Time features
    # -----------------------
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")

    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds()

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    # -----------------------
    # User velocity features
    # -----------------------
    df = df.sort_values(["user_id", "purchase_time"])

    df["tx_count_uder_id_1h"] = (
        df.groupby("user_id")
          .rolling("1h", on="purchase_time")["purchase_time"]
          .count()
          .to_numpy()
    )

    df["tx_count_uder_id_24h"] = (
        df.groupby("user_id")
          .rolling("24h", on="purchase_time")["purchase_time"]
          .count()
          .to_numpy()
    )

    df[["tx_count_uder_id_1h", "tx_count_uder_id_24h"]] = (
        df[["tx_count_uder_id_1h", "tx_count_uder_id_24h"]].fillna(0)
    )

    # Restore original order
    df = df.loc[original_index]

    # -----------------------
    # Global frequency features
    # -----------------------
    df["device_id_count"] = df.groupby(
        "device_id")["device_id"].transform("count")
    df["ip_address_count"] = df.groupby(
        "ip_address")["ip_address"].transform("count")
    df["user_total_transactions"] = df.groupby(
        "user_id")["user_id"].transform("count")

    return df
