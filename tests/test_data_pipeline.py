import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from fraud_detection.core.settings import settings
from fraud_detection.data.cleaning import DataCleaning
from fraud_detection.features.pipeline import build_feature_pipeline
import fraud_detection.data.ip_geolocation as ip


def test_pipeline():
    # -----------------------------
    # 1. Load a robust sample of raw data
    # -----------------------------
    # We need enough rows so that:
    # - Stratified split can put Class 1 in both Train and Test.
    # - SMOTE has enough "neighbors" to generate synthetic data.
    fraud_df = pd.DataFrame({
        "user_id": range(1, 11),
        "signup_time": ["2020-01-01 08:00:00"] * 10,
        "purchase_time": ["2020-01-05 08:00:00"] * 10,
        "purchase_value": [100, 150, 200, 250, 300, 100, 150, 200, 250, 300],
        "device_id": [f"d{i}" for i in range(10)],
        "source": ["web", "app"] * 5,
        "browser": ["Chrome", "Firefox", "Safari", "Edge", "Opera"] * 2,
        "sex": ["M", "F"] * 5,
        "age": [30, 25, 40, 35, 22, 31, 28, 45, 33, 24],
        "ip_address": [3232235776 + i for i in range(10)],
        "class": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # 4 Fraud (1), 6 Legit (0)
        "country": ["US", "UK", "US", "US", "UK", "US", "UK", "US", "US", "UK"]
    })

    ip_df = pd.DataFrame({
        "lower_bound_ip_address": [3232235776],
        "upper_bound_ip_address": [3232235800],
        "country": ["US"]
    })

    # -----------------------------
    # 2. Clean fraud data
    # -----------------------------
    cleaner = DataCleaning(
        drop_duplicates=True,
        duplicate_subset=["user_id", "purchase_time", "purchase_value"],
        strip_strings=True,
        protected_string_columns=["user_id", "device_id", "ip_address"],
        empty_string_as_nan=True,
        datetime_columns=["signup_time", "purchase_time"],
        numeric_columns=["purchase_value", "age"],
        verbose=False
    )
    cleaned_df = cleaner.clean(fraud_df)

    # -----------------------------
    # 3. Clean IP reference
    # -----------------------------
    ip_country_df = ip.clean_ip_country_table(ip_df)
    cleaned_df = ip.normalize_ip_column(cleaned_df, ip_col="ip_address")
    df = ip.map_ip_to_country(cleaned_df, ip_country_df)

    # -----------------------------
    # 4. Prepare features and target
    # -----------------------------
    FEATURES = settings.get("features")
    TARGET = FEATURES["target"]
    NUM_COLS = FEATURES["numeric"]
    CAT_COLS = FEATURES["categorical"]

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # -----------------------------
    # 5. Train/test split (Stratify now works with 4 fraud samples)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # -----------------------------
    # 6. Feature pipeline
    # -----------------------------
    feature_pipeline = build_feature_pipeline(NUM_COLS, CAT_COLS)
    X_train_transformed = feature_pipeline.fit_transform(X_train)
    X_test_transformed = feature_pipeline.transform(X_test)

    # -----------------------------
    # 7. SMOTE on training data
    # -----------------------------
    # NOTE: Set k_neighbors=1 because our training set is still very small
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed, y_train
    )

    print("\n--- Class Distribution AFTER SMOTE ---")
    print(y_train_resampled.value_counts(normalize=True).map("{:.2%}".format))

    # -----------------------------
    # 8. Assertions (Important for Testing)
    # -----------------------------
    assert y_train_resampled.value_counts()[0] == y_train_resampled.value_counts()[
        1], "SMOTE failed to balance classes"
    assert len(X_test_transformed) > 0, "Test set should not be empty"

    print("\n[SUCCESS] Pipeline test passed.")


if __name__ == "__main__":
    test_pipeline()
