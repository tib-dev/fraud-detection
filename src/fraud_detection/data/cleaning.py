import pandas as pd
import numpy as np
from typing import Optional, List


class DataCleaning:
    """
    Performs deterministic, rule-based data cleaning.

    This step handles obvious data quality issues before
    feature engineering or modeling.
    """

    def __init__(
        self,
        drop_duplicates: bool = True,
        strip_strings: bool = True,
        empty_string_as_nan: bool = True,
        datetime_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.drop_duplicates = drop_duplicates
        self.strip_strings = strip_strings
        self.empty_string_as_nan = empty_string_as_nan
        self.datetime_columns = datetime_columns or []
        self.numeric_columns = numeric_columns or []
        self.verbose = verbose

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data cleaning steps to the input DataFrame.
        """
        df = df.copy()

        if self.verbose:
            print("Starting data cleaning...")
            print(f"Initial shape: {df.shape}\n")

        if self.strip_strings:
            self._strip_string_values(df)

        if self.empty_string_as_nan:
            self._replace_empty_strings(df)

        self._convert_datetime_columns(df)
        self._convert_numeric_columns(df)

        if self.drop_duplicates:
            self._drop_duplicates(df)

        if self.verbose:
            print(f"\nFinal shape after cleaning: {df.shape}")
            print("Data cleaning completed.\n")

        return df

    def _strip_string_values(self, df: pd.DataFrame) -> None:
        string_cols = df.select_dtypes(include="object").columns

        if len(string_cols) == 0:
            if self.verbose:
                print("No string columns to strip.")
            return

        for col in string_cols:
            df[col] = df[col].str.strip()

        if self.verbose:
            print(
                f"Stripped whitespace from {len(string_cols)} string columns.")

    def _replace_empty_strings(self, df: pd.DataFrame) -> None:
        empty_count = (df == "").sum().sum()

        if empty_count == 0:
            if self.verbose:
                print("No empty strings found.")
            return

        df.replace("", np.nan, inplace=True)

        if self.verbose:
            print(f"Replaced {empty_count} empty strings with NaN.")

    def _convert_datetime_columns(self, df: pd.DataFrame) -> None:
        for col in self.datetime_columns:
            if col not in df.columns:
                if self.verbose:
                    print(f"Datetime column '{col}' not found.")
                continue

            before_na = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after_na = df[col].isna().sum()

            if self.verbose:
                print(
                    f"Converted '{col}' to datetime "
                    f"(NaN before: {before_na}, after: {after_na})."
                )

    def _convert_numeric_columns(self, df: pd.DataFrame) -> None:
        for col in self.numeric_columns:
            if col not in df.columns:
                if self.verbose:
                    print(f"Numeric column '{col}' not found.")
                continue

            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            after_na = df[col].isna().sum()

            if self.verbose:
                print(
                    f"Converted '{col}' to numeric "
                    f"(NaN before: {before_na}, after: {after_na})."
                )

    def _drop_duplicates(self, df: pd.DataFrame) -> None:
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)

        if before == after:
            if self.verbose:
                print("No duplicate rows found.")
        else:
            if self.verbose:
                print(f"Removed {before - after} duplicate rows.")


def normalize_ip_range(
    df,
    lower_col="lower_bound_ip_address",
    upper_col="upper_bound_ip_address"
):
    df = df.copy()

    # Convert to numeric safely
    df[lower_col] = pd.to_numeric(df[lower_col], errors="coerce")
    df[upper_col] = pd.to_numeric(df[upper_col], errors="coerce")

    # Report missing values
    missing_lower = df[lower_col].isna().sum()
    missing_upper = df[upper_col].isna().sum()

    if missing_lower == 0 and missing_upper == 0:
        print("No missing IP range values found.")
    else:
        print(
            f"Missing IP ranges → "
            f"lower: {missing_lower}, upper: {missing_upper}"
        )

    # Drop invalid rows (cannot be used for range matching)
    before = len(df)
    df = df.dropna(subset=[lower_col, upper_col])
    after = len(df)

    if before == after:
        print("No invalid IP range rows removed.")
    else:
        print(f"Removed {before - after} invalid IP range rows.")

    # Convert to integer
    df[lower_col] = df[lower_col].astype(np.int64)
    df[upper_col] = df[upper_col].astype(np.int64)

    print("IP range columns converted to int64.")

    return df


def normalize_ip_address(
    df,
    ip_col="ip_address",
    verbose=True
):
    df = df.copy()

    if ip_col not in df.columns:
        raise ValueError(f"Column '{ip_col}' not found")

    # Convert to numeric safely
    before_na = df[ip_col].isna().sum()
    df[ip_col] = pd.to_numeric(df[ip_col], errors="coerce")
    after_na = df[ip_col].isna().sum()

    if verbose:
        print(
            f"IP column '{ip_col}' numeric conversion "
            f"(NaN before: {before_na}, after: {after_na})"
        )

    # Drop invalid IPs
    before = len(df)
    df = df.dropna(subset=[ip_col])
    after = len(df)

    if verbose:
        if before == after:
            print("No invalid IP rows removed.")
        else:
            print(f"Removed {before - after} rows with invalid IPs.")

    # Convert to int64
    df[ip_col] = df[ip_col].astype(np.int64)

    if verbose:
        print(f"'{ip_col}' converted to int64.")

    return df
