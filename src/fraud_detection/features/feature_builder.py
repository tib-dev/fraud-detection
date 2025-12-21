import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FraudFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_dict = {}

    def extract_time_features(self, df):
        """Extracts temporal signals from timestamps."""
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # 1. Time-to-Purchase: Crucial for spotting instant-purchase bots
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']).dt.total_seconds()

        # 2. Temporal context: When do fraudsters strike?
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        return df

    def extract_velocity_features(self, df):
        """Calculates how many times an ID appears (Frequency/Velocity)."""
        # 3. Device & IP frequency: Identifying botnets and account takeovers
        df['device_id_count'] = df.groupby(
            'device_id')['device_id'].transform('count')
        df['ip_address_count'] = df.groupby(
            'ip_address')['ip_address'].transform('count')
        return df

    def handle_categorical_features(self, df, categories=['source', 'browser', 'sex', 'country']):
        """Encodes text data into numeric format."""
        for col in categories:
            if col in df.columns:
                le = LabelEncoder()
                # Fill missing countries as 'Unknown'
                df[col] = df[col].fillna('Unknown').astype(str)
                df[col] = le.fit_transform(df[col])
                self.le_dict[col] = le
        return df

    def scale_numerical_data(self, df, num_cols):
        """Normalizes ranges so large numbers don't dominate the model."""
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        return df

    def prepare_final_df(self, df):
        """Drops raw columns that models cannot process."""
        # We drop raw IDs and timestamps because they are strings/objects
        # The logic is already captured in our engineered features
        drop_cols = [
            'user_id', 'device_id', 'signup_time', 'purchase_time',
            'ip_address', 'lower_bound_ip_address', 'upper_bound_ip_address'
        ]
        # Only drop if they exist to prevent errors
        existing_drop_cols = [c for c in drop_cols if c in df.columns]
        return df.drop(columns=existing_drop_cols)

# --- EXECUTION ---
# engineer = FraudFeatureEngineer()
# df = engineer.extract_time_features(df)
# df = engineer.extract_velocity_features(df)
# df = engineer.handle_categorical_features(df)
#
# List of columns to scale for distance-based models (like Logistic Regression)
# num_cols = ['purchase_value', 'age', 'time_since_signup', 'device_id_count', 'ip_address_count']
# df = engineer.scale_numerical_data(df, num_cols)
#
# df_final = engineer.prepare_final_df(df)
