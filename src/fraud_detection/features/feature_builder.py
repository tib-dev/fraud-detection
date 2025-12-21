import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    from sklearn.utils import resample
    HAS_SMOTE = False


class FraudFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_dict = {}

    # --- 1. IP-TO-INTEGER & COUNTRY MERGE ---
    def merge_with_country(self, fraud_df, country_df):
        f_df = fraud_df.copy()
        c_df = country_df.copy()

        # Convert IP to numeric and sort (Essential for merge_asof)
        f_df['ip_int'] = pd.to_numeric(
            f_df['ip_address'], errors="coerce").fillna(0).astype(np.int64)
        c_df['lower_bound_ip_address'] = pd.to_numeric(
            c_df['lower_bound_ip_address'], errors="coerce").fillna(0).astype(np.int64)
        c_df['upper_bound_ip_address'] = pd.to_numeric(
            c_df['upper_bound_ip_address'], errors="coerce").fillna(0).astype(np.int64)

        f_df = f_df.sort_values('ip_int')
        c_df = c_df.sort_values('lower_bound_ip_address')

        # Merge based on lower bound
        merged = pd.merge_asof(f_df, c_df, left_on='ip_int',
                               right_on='lower_bound_ip_address')

        # Validation: check if IP is within the upper bound
        merged['country'] = np.where(merged['ip_int'] <= merged['upper_bound_ip_address'],
                                     merged['country'], 'Unknown')
        print(f"✅ Country Merge Complete.")
        return merged

    # --- 2. TIME & VELOCITY FEATURES (RESTORED) ---
    def extract_time_features(self, df):
        df = df.copy()
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        return df

    def extract_velocity_features(self, df):
        df = df.copy()
        df['device_id_count'] = df.groupby(
            'device_id')['device_id'].transform('count')
        df['ip_address_count'] = df.groupby(
            'ip_address')['ip_address'].transform('count')
        return df

    # --- 3. SCALING & ENCODING ---
    def preprocess_pipeline(self, df, num_cols):
        df = df.copy()
        # Categorical Encoding
        for col in ['source', 'browser', 'sex', 'country']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(
                    df[col].fillna('Unknown').astype(str))
                self.le_dict[col] = le
        # StandardScaler
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        print(f"✅ Preprocessing: Scaling and Encoding Complete.")
        return df

    # --- 4. SMOTE (TRAIN ONLY) ---
    def split_and_resample(self, df, target='class'):
        drop_cols = [target, 'user_id', 'device_id',
                     'signup_time', 'purchase_time', 'ip_address', 'ip_int']
        X = df.drop(
            columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        y = df[target].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        print("\n--- Distribution BEFORE SMOTE ---")
        print(y_train.value_counts(normalize=True).map('{:.2%}'.format))

        if HAS_SMOTE:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            # Undersampling Fallback
            train_data = pd.concat([X_train, y_train], axis=1)
            fraud = train_data[train_data[target] == 1]
            legit = train_data[train_data[target] == 0].sample(
                len(fraud), random_state=42)
            resampled = pd.concat([fraud, legit])
            X_train_res, y_train_res = resampled.drop(
                columns=[target]), resampled[target]

        print("\n--- Distribution AFTER SMOTE ---")
        print(y_train_res.value_counts(normalize=True).map('{:.2%}'.format))
        return X_train_res, X_test, y_train_res, y_test

    def verify(self, df):
        """Mathematical verification of scaling."""
        mean_val = df['purchase_value'].mean()
        assert abs(mean_val) < 1e-10, "Scaling failed!"
        print("🚀 Pipeline Verification Passed.")
