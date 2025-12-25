import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # -----------------------
        # Time features
        # -----------------------
        df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
        df["purchase_time"] = pd.to_datetime(
            df["purchase_time"], errors="coerce")

        df["time_since_signup"] = (
            df["purchase_time"] - df["signup_time"]
        ).dt.total_seconds()

        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek

        # -----------------------
        # User velocity (FIXED)
        # -----------------------
        df = df.sort_values(["user_id", "purchase_time"])

        df["tx_count_1h"] = (
            df.groupby("user_id")
            .rolling("1h", on="purchase_time")["purchase_time"]
            .count()
            .to_numpy()
        )

        df["tx_count_24h"] = (
            df.groupby("user_id")
            .rolling("24h", on="purchase_time")["purchase_time"]
            .count()
            .to_numpy()
        )

        df[["tx_count_1h", "tx_count_24h"]] = (
            df[["tx_count_1h", "tx_count_24h"]].fillna(0)
        )

        # -----------------------
        # Global counts
        # -----------------------
        df["device_id_count"] = df.groupby(
            "device_id")["device_id"].transform("count")
        df["ip_address_count"] = df.groupby(
            "ip_address")["ip_address"].transform("count")

        # -----------------------
        # Drop raw identifiers
        # -----------------------
        df = df.drop(
            columns=["user_id", "device_id", "signup_time",
                     "purchase_time", "ip_address"],
            errors="ignore"
        )

        return df
