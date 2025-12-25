import pandas as pd


def convert_dtype(
    df,
    columns,
    dtype,
    errors="coerce",
    fillna=None,
    dropna=False,
    verbose=True
):
    """
    Safely convert one or more columns to a target dtype.

    Parameters
    ----------
    df : pd.DataFrame
    columns : str or list[str]
    dtype : numpy dtype or str
    errors : {'raise', 'coerce', 'ignore'}
    fillna : scalar or None
    dropna : bool
    verbose : bool
    """

    df = df.copy()

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

        before_na = df[col].isna().sum()

        df[col] = pd.to_numeric(df[col], errors=errors)

        after_na = df[col].isna().sum()

        if fillna is not None:
            df[col] = df[col].fillna(fillna)

        dropped = 0
        if dropna:
            before = len(df)
            df = df.dropna(subset=[col])
            dropped = before - len(df)

        df[col] = df[col].astype(dtype)

        if verbose:
            print(
                f"[{col}] → {dtype} | "
                f"NaN before: {before_na}, after: {after_na}, "
                f"rows dropped: {dropped}"
            )

    return df
