import pandas as pd
import numpy as np
from typing import Optional, List


class DataCleaning:
    """
    Deterministic, rule-based data cleaning for fraud datasets.

    IMPORTANT:
    - This class does NOT perform imputation.
    - Missingness is preserved for downstream feature engineering.
    """

    def __init__(
        self,
        drop_duplicates: bool = False,
        duplicate_subset: Optional[List[str]] = None,
        strip_strings: bool = True,
        protected_string_columns: Optional[List[str]] = None,
        empty_string_as_nan: bool = True,
        datetime_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.drop_duplicates = drop_duplicates
        self.duplicate_subset = duplicate_subset
        self.strip_strings = strip_strings
        self.protected_string_columns = set(protected_string_columns or [])
        self.empty_string_as_nan = empty_string_as_nan
        self.datetime_columns = datetime_columns or []
        self.numeric_columns = numeric_columns or []
        self.verbose = verbose

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.verbose:
            print(f"Starting data cleaning | rows={len(df)}")

        if self.strip_strings:
            self._strip_string_values(df)

        if self.empty_string_as_nan:
            self._replace_empty_strings(df)

        self._convert_datetime_columns(df)
        self._convert_numeric_columns(df)

        if self.drop_duplicates:
            self._drop_duplicates(df)

        if self.verbose:
            print(f"Finished data cleaning | rows={len(df)}\n")

        return df

    def _strip_string_values(self, df: pd.DataFrame) -> None:
        string_cols = [
            c for c in df.select_dtypes(include="object").columns
            if c not in self.protected_string_columns
        ]

        for col in string_cols:
            df[col] = df[col].str.strip()

        if self.verbose and string_cols:
            print(
                f"Stripped whitespace from {len(string_cols)} string columns.")

    def _replace_empty_strings(self, df: pd.DataFrame) -> None:
        string_cols = df.select_dtypes(include="object").columns
        empty_mask = df[string_cols] == ""
        empty_count = empty_mask.sum().sum()

        if empty_count > 0:
            df[string_cols] = df[string_cols].replace("", np.nan)
            if self.verbose:
                print(f"Replaced {empty_count} empty strings with NaN.")

    def _convert_datetime_columns(self, df: pd.DataFrame) -> None:
        for col in self.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    def _convert_numeric_columns(self, df: pd.DataFrame) -> None:
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def _drop_duplicates(self, df: pd.DataFrame) -> None:
        before = len(df)
        df.drop_duplicates(subset=self.duplicate_subset, inplace=True)
        removed = before - len(df)

        if self.verbose:
            print(f"Removed {removed} duplicate rows.")
