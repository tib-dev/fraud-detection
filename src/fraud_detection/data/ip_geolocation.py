import pandas as pd
import numpy as np
from bisect import bisect_right

MAX_IPV4 = 2**32 - 1


def clean_ip_country_table(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean and validate IP-to-country reference table.
    Drops invalid ranges and enforces IPv4 compliance.
    """
    df = df.copy()

    # Required columns
    required_cols = ["lower_bound_ip_address",
                     "upper_bound_ip_address", "country"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Coerce to numeric
    df["lower_bound_ip_address"] = pd.to_numeric(
        df["lower_bound_ip_address"], errors="coerce")
    df["upper_bound_ip_address"] = pd.to_numeric(
        df["upper_bound_ip_address"], errors="coerce")

    before = len(df)

    # Drop invalid ranges
    df = df.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"])
    df = df[
        (df["lower_bound_ip_address"] >= 0) &
        (df["upper_bound_ip_address"] <= MAX_IPV4) &
        (df["lower_bound_ip_address"] <= df["upper_bound_ip_address"])
    ]

    # Normalize country
    df["country"] = df["country"].astype(str).str.strip().replace(
        {"": "Unknown", "nan": "Unknown"})

    # Deduplicate exact ranges
    df = df.drop_duplicates(
        subset=["lower_bound_ip_address", "upper_bound_ip_address", "country"])

    # Sort for interval search
    df = df.sort_values("lower_bound_ip_address").reset_index(drop=True)

    if verbose:
        print(
            f"[IP TABLE CLEAN] {before - len(df)} invalid rows removed, final size {len(df)}")

    return df


def normalize_ip_column(df: pd.DataFrame, ip_col: str = "ip_address", out_col: str = "ip_int", verbose: bool = True) -> pd.DataFrame:
    """
    Normalize transaction IP addresses to integer form.
    Drops invalid IPs outside 0 - 2^32-1.
    """
    df = df.copy()
    df[out_col] = pd.to_numeric(df[ip_col], errors="coerce")
    before = len(df)

    df = df.dropna(subset=[out_col])
    df = df[(df[out_col] >= 0) & (df[out_col] <= MAX_IPV4)]
    df[out_col] = df[out_col].astype(np.int64)

    if verbose:
        print(f"[IP NORMALIZATION] Removed {before - len(df)} invalid IP rows")

    return df


def map_ip_to_country(df: pd.DataFrame, country_df: pd.DataFrame, ip_col: str = "ip_address", verbose: bool = True) -> pd.DataFrame:
    """
    Map normalized IPs to countries using vectorized interval search.
    Faster and exact for large datasets.
    """
    df = df.copy()
    country_df = country_df.copy()

    # Precompute list of lower bounds for search
    lower_bounds = country_df["lower_bound_ip_address"].values
    upper_bounds = country_df["upper_bound_ip_address"].values
    countries = country_df["country"].values

    # Function to map single IP to country
    def ip_to_country(ip):
        idx = bisect_right(lower_bounds, ip) - 1
        if idx >= 0 and ip <= upper_bounds[idx]:
            return countries[idx]
        return "Unknown"

    # Apply mapping vectorized (can use multiprocessing for very large data)
    df["country"] = df[ip_col].apply(ip_to_country)

    if verbose:
        coverage = (df["country"] != "Unknown").mean() * 100
        print(f"[IP MERGE] Country coverage: {coverage:.2f}%")

    return df
