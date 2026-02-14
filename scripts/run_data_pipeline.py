import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import fraud_detection.data.ip_geolocation as ip
from fraud_detection.data.loader import DataHandler
from fraud_detection.core.settings import settings
from fraud_detection.data.cleaning import DataCleaning
from fraud_detection.features.preprocessing import build_preprocessing_pipeline
from fraud_detection.features.custom_features import add_fraud_features


def run_pipeline():
    # -----------------------------
    # 0. Load raw data
    # -----------------------------

    fraud_df = DataHandler.from_registry(
        "DATA", "raw_dir", "Fraud_Data.csv").load()
    ip_df = DataHandler.from_registry(
        "DATA", "raw_dir", "IpAddress_to_Country.csv").load()

    # -----------------------------
    # 1. Clean fraud data
    # -----------------------------
    datetime_cols = ["signup_time", "purchase_time"]
    numeric_cols = ["purchase_value", "age"]
    protected_cols = ["user_id", "device_id", "ip_address"]  # Do not strip

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
    # 2. Clean reference table & map IPs
    # -----------------------------
    ip_country_df = ip.clean_ip_country_table(ip_df)
    fraud_df = ip.normalize_ip_column(cleaned_df, ip_col="ip_address")
    df = ip.map_ip_to_country(fraud_df, ip_country_df)
    print(f"[INFO] IP mapping completed. Dataset shape: {df.shape}")

    # -----------------------------
    # 3. Add engineered features
    # -----------------------------
    df_features = add_fraud_features(df)

    # -----------------------------
    # 4. Split features & target
    # -----------------------------
    FEATURES = settings.get("features")
    TARGET = FEATURES["target"]
    NUM_COLS = FEATURES["numeric"]
    CAT_COLS = FEATURES["categorical"]

    X = df_features.drop(columns=[TARGET])
    y = df_features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n--- Class Distribution BEFORE SMOTE (Train) ---")
    print(y_train.value_counts(normalize=True).map("{:.2%}".format))

    # -----------------------------
    # 5. Build & apply preprocessing pipeline
    # -----------------------------
    preprocessor = build_preprocessing_pipeline(NUM_COLS, CAT_COLS)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract feature names
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_train_df[TARGET] = y_train.reset_index(drop=True)

    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    X_test_df[TARGET] = y_test.reset_index(drop=True)

    # -----------------------------
    # 6. Apply SMOTE (train only)
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed, y_train)

    X_train_res_df = pd.DataFrame(X_train_resampled, columns=feature_names)
    X_train_res_df[TARGET] = y_train_resampled.reset_index(drop=True)

    print("\n--- Class Distribution AFTER SMOTE ---")
    print(y_train_resampled.value_counts(normalize=True).map("{:.2%}".format))

    # -----------------------------
    # 7. Save processed datasets
    # -----------------------------
    train_original_df = pd.DataFrame(
        X_train_transformed, columns=feature_names)


    train_original_df[TARGET] = y_train.reset_index(drop=True)

    DataHandler.from_registry(
        section="DATA",
        path_key="processed_dir",
        filename="train_original.parquet"
    ).save(train_original_df)

    DataHandler.from_registry(
        section="DATA",
        path_key="processed_dir",
        filename="train_resampled.parquet"
    ).save(X_train_res_df)

    DataHandler.from_registry(
        section="DATA",
        path_key="processed_dir",
        filename="test_original.parquet"
    ).save(X_test_df)

    print("[INFO] Preprocessing and SMOTE complete. Data saved successfully.")


if __name__ == "__main__":
    run_pipeline()
