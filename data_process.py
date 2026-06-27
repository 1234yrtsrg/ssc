# -*- coding: utf-8 -*-
"""
Data processing script for the empirical asset pricing ML project.

This script creates the cleaned stock-level panel used by all prediction models.
It only handles data cleaning and characteristic preprocessing. It does not train
any machine learning model.
"""
import os
os.chdir(r"D:\empirical-asset-pricing-ml\scripts")


from config_global import *

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd




def check_required_columns(df, required_cols):
    """Stop the script if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"The raw data are missing these columns: {missing}")


def cross_sectional_rank(df, date_col, firmcols):
    """
    Rank firm characteristics within each month.

    The transformation maps each characteristic to approximately [-0.5, 0.5].
    This makes variables comparable across months and reduces the influence of
    extreme values.
    """
    ranked = df.groupby(date_col)[firmcols].transform(
        lambda x: (x.rank(pct=True) * 2 - 1) / 2
    )
    return ranked


def main():
    print("Loading raw data from:", RAW_DATA_PATH)
    crsp = pd.read_parquet(RAW_DATA_PATH)

    # Keep only the columns needed for this project.
    base_cols = [
        DATE_COL,
        ID_COL,
        RET_COL,
        PRC_COL,
        EXCHCD_COL,
        SHRCD_COL,
        ME_COL,
        SIC_COL,
        SIZE_GRP_COL,
    ]
    required_cols = base_cols + FIRM_COLS
    check_required_columns(crsp, required_cols)
    crsp = crsp[required_cols].copy()

    # Convert dates to month-end. Monthly alignment is crucial in return prediction.
    crsp[DATE_COL] = pd.to_datetime(crsp[DATE_COL], format="%Y-%m-%d") + MonthEnd(0)

    # Keep observations with a valid stock identifier.
    crsp = crsp.loc[crsp[ID_COL].notna()].copy()
    crsp[ID_COL] = crsp[ID_COL].astype("int32")

    # Standard CRSP universe filter: NYSE/AMEX/NASDAQ common shares.
    crsp = crsp.loc[
        crsp[EXCHCD_COL].isin([1, 2, 3]) & crsp[SHRCD_COL].isin([10, 11])
    ].copy()

    # CRSP price can be negative because of bid-ask average conventions.
    # We use absolute price and drop very low-price stocks.
    crsp[PRC_COL] = crsp[PRC_COL].abs()
    crsp = crsp.loc[crsp[PRC_COL] >= 1].copy()

    # Basic panel cleaning.
    crsp = crsp.sort_values([ID_COL, DATE_COL])
    crsp = crsp.loc[crsp[DATE_COL].dt.year >= START_YEAR].copy()
    crsp = crsp.drop_duplicates([ID_COL, DATE_COL], keep="first")

    # Keep market equity for portfolio weighting.
    # The name LME_COL is kept for compatibility with portfolio scripts.
    crsp[LME_COL] = crsp[ME_COL]

    # Keep the final columns used downstream.
    keep_cols = [DATE_COL, ID_COL, RET_COL, LME_COL, SIZE_GRP_COL, PRC_COL] + FIRM_COLS
    crsp = crsp[keep_cols].copy()

    # Express returns in percent, following many empirical asset pricing papers.
    crsp[RET_COL] = crsp[RET_COL] * 100

    # The target return and portfolio weight must be non-missing.
    crsp = crsp.dropna(subset=[RET_COL, LME_COL]).copy()

    # Rank-normalize firm characteristics month by month.
    print("Applying monthly cross-sectional rank transformation...")
    crsp[FIRM_COLS] = cross_sectional_rank(crsp, DATE_COL, FIRM_COLS)

    # Fill missing ranked characteristics with monthly medians.
    # If a variable is missing for an entire month, the median is also missing;
    # the final fillna(0.0) assigns a neutral value after rank normalization.
    crsp[FIRM_COLS] = crsp.groupby(DATE_COL)[FIRM_COLS].transform(
        lambda x: x.fillna(x.median())
    )
    remaining_na = int(crsp[FIRM_COLS].isna().sum().sum())
    if remaining_na > 0:
        print(f"Remaining missing characteristic values after median fill: {remaining_na}")
        crsp[FIRM_COLS] = crsp[FIRM_COLS].fillna(0.0)

    # Save the cleaned panel.
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    crsp.to_parquet(DATA_CLEAN_PATH)

    print("Saved cleaned panel to:", DATA_CLEAN_PATH)
    print("Final panel shape:", crsp.shape)
    print("Sample period:", crsp[DATE_COL].min(), "to", crsp[DATE_COL].max())
    print("Number of firm characteristics:", len(FIRM_COLS))


if __name__ == "__main__":
    main()
