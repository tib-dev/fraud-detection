import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from fraud_detection.data.loader import DataHandler
from fraud_detection.core.settings import settings
from fraud_detection.data.cleaning import DataCleaning
from fraud_detection.features.pipeline import build_feature_pipeline
import fraud_detection.data.ip_geolocation as ip


def run_pipeline():
    # -----------------------------
    # 1. Load raw data
    # -----------------------------
    fraud_df = DataHandler.from_registry(
        "DATA", "raw_dir", "Fraud_Data.csv").load()
    ip_df = DataHandler.from_registry(
        "DATA", "raw_dir", "IpAddress_to_Country.csv").load()
    print(f"[INFO] Loaded raw fraud data: {fraud_df.shape}")
    print(f"[INFO] Loaded IP reference data: {ip_df.shape}")

    # -----------------------------
    # 2. Clean fraud data
    # -----------------------------
    datetime_cols = ["signup_time", "purchase_time"]
    numeric_cols = ["purchase_value", "age"]
    protected_cols = ["user_id", "device_id", "ip_address"]

    cleaner = DataCleaning(
        drop_duplicates=True,
        duplicate_subset=["user_id", "purchase_time", "purchase_value"],
        strip_strings=True,
        protected_string_columns=protected_cols,
        empty_string_as_nan=True,
        datetime_columns=datetime_cols,
        numeric_columns=numeric_cols,
        verbose=True
    )
    cleaned_df = cleaner.clean(fraud_df)
    print(f"[INFO] Cleaned fraud data shape: {cleaned_df.shape}")

    # -----------------------------
    # 3. Clean and normalize IP reference table
    # -----------------------------
    ip_country_df = ip.clean_ip_country_table(ip_df)
    cleaned_df = ip.normalize_ip_column(cleaned_df, ip_col="ip_address")
    df = ip.map_ip_to_country(cleaned_df, ip_country_df)
    print(f"[INFO] IP mapping completed. Dataset shape: {df.shape}")

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
    # 5. Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(
        f"[INFO] Split train/test: X_train={X_train.shape}, X_test={X_test.shape}")

    # -----------------------------
    # 6. Build and apply feature pipeline
    # -----------------------------
    feature_pipeline = build_feature_pipeline(NUM_COLS, CAT_COLS)
    X_train_transformed = feature_pipeline.fit_transform(X_train)
    X_test_transformed = feature_pipeline.transform(X_test)

    # Get transformed feature names
    transformers = feature_pipeline.named_steps['preprocessing'].transformers_
    num_cols = transformers[0][2]
    cat_cols = transformers[1][1].named_steps['encoder'].get_feature_names_out(
        CAT_COLS)
    feature_names = list(num_cols) + list(cat_cols)

    # -----------------------------
    # 7. Save pre-SMOTE training data
    # -----------------------------
    train_original_df = pd.DataFrame(
        X_train_transformed, columns=feature_names)
    train_original_df[TARGET] = y_train.reset_index(drop=True)

    train_original_handler = DataHandler.from_registry(
        section="DATA",
        path_key="processed_dir",
        filename="train_original.parquet"
    )
    train_original_handler.save(train_original_df)
    print("[INFO] Pre-SMOTE training data saved.")

    # Save test set (unchanged)
    test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    test_df[TARGET] = y_test.reset_index(drop=True)

    test_handler = DataHandler.from_registry(
        section="DATA",
        path_key="processed_dir",
        filename="test_original.parquet"
    )
    test_handler.save(test_df)
    print("[INFO] Test data saved.")

    # -----------------------------
    # 8. Apply SMOTE
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed, y_train)

    print("\n--- Class Distribution BEFORE SMOTE ---")
    print(y_train.value_counts(normalize=True).map("{:.2%}".format))

    print("\n--- Class Distribution AFTER SMOTE ---")
    print(y_train_resampled.value_counts(normalize=True).map("{:.2%}".format))

    # -----------------------------
    # 9. Save post-SMOTE training data
    # -----------------------------
    train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_names)
    train_resampled_df[TARGET] = y_train_resampled.reset_index(drop=True)

    train_resampled_handler = DataHandler.from_registry(
        section="DATA",
        path_key="processed_dir",
        filename="train_resampled.parquet"
    )
    train_resampled_handler.save(train_resampled_df)
    print("[INFO] Post-SMOTE training data saved.")


if __name__ == "__main__":
    run_pipeline()
