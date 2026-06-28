# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:44:26 2026

@author: HUAWEI
"""

# -*- coding: utf-8 -*-
"""
Data processing script for the empirical asset pricing ML project.
This script creates the cleaned stock-level panel used by all prediction models.
It only handles data cleaning and characteristic preprocessing. It does not train
any machine learning model.
"""
import os
os.chdir("/Users/howardhua-pc/Desktop/empirical-asset-pring-ml/scripts")
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
    ranked = df.groupby(date_col)[firmcols].transform(
        lambda x: (x.rank(pct=True) * 2 - 1) / 2
    )
    return ranked

def monthly_winsorize(df, date_col, firmcols, lower=0.01, upper=0.99):
    """按月对每个特征做上下1%缩尾处理."""
    df = df.copy()
    for col in firmcols:
        low  = df.groupby(date_col)[col].transform(lambda x: x.quantile(lower))
        high = df.groupby(date_col)[col].transform(lambda x: x.quantile(upper))
        df[col] = df[col].clip(lower=low, upper=high)
    return df

def main():
    print("Loading raw data from:", RAW_DATA_PATH)
    crsp = pd.read_parquet(RAW_DATA_PATH)
    base_cols = [
        DATE_COL, ID_COL, RET_COL, PRC_COL,
        EXCHCD_COL, SHRCD_COL, ME_COL, SIC_COL, SIZE_GRP_COL,
    ]
    required_cols = base_cols + FIRM_COLS
    check_required_columns(crsp, required_cols)
    crsp = crsp[required_cols].copy()
    crsp[DATE_COL] = pd.to_datetime(crsp[DATE_COL], format="%Y-%m-%d") + MonthEnd(0)
    crsp = crsp.loc[crsp[ID_COL].notna()].copy()
    crsp[ID_COL] = crsp[ID_COL].astype("int32")
    crsp = crsp.loc[
        crsp[EXCHCD_COL].isin([1, 2, 3]) & crsp[SHRCD_COL].isin([10, 11])
    ].copy()
    crsp[PRC_COL] = crsp[PRC_COL].abs()
    crsp = crsp.loc[crsp[PRC_COL] >= 1].copy()
    crsp = crsp.sort_values([ID_COL, DATE_COL])
    crsp = crsp.loc[crsp[DATE_COL].dt.year >= START_YEAR].copy()
    crsp = crsp.drop_duplicates([ID_COL, DATE_COL], keep="first")
    crsp[LME_COL] = crsp[ME_COL]
    keep_cols = [DATE_COL, ID_COL, RET_COL, LME_COL, SIZE_GRP_COL, PRC_COL] + FIRM_COLS
    crsp = crsp[keep_cols].copy()
    crsp[RET_COL] = crsp[RET_COL] * 100
    crsp = crsp.dropna(subset=[RET_COL, LME_COL]).copy()

    # ── 新增：winsorize后存盘（rank之前）──
    print("Applying monthly winsorization (1%-99%)...")
    crsp = monthly_winsorize(crsp, DATE_COL, FIRM_COLS)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    winsorize_path = DATA_PROCESSED_DIR / "panel_winsorize_only.parquet"
    crsp.to_parquet(winsorize_path)
    print("Saved winsorize-only panel to:", winsorize_path)
    # ─────────────────────────────────────

    print("Applying monthly cross-sectional rank transformation...")
    crsp[FIRM_COLS] = cross_sectional_rank(crsp, DATE_COL, FIRM_COLS)
    crsp[FIRM_COLS] = crsp.groupby(DATE_COL)[FIRM_COLS].transform(
        lambda x: x.fillna(x.median())
    )
    remaining_na = int(crsp[FIRM_COLS].isna().sum().sum())
    if remaining_na > 0:
        print(f"Remaining missing characteristic values after median fill: {remaining_na}")
        crsp[FIRM_COLS] = crsp[FIRM_COLS].fillna(0.0)
    crsp.to_parquet(DATA_CLEAN_PATH)
    print("Saved cleaned panel to:", DATA_CLEAN_PATH)
    print("Final panel shape:", crsp.shape)
    print("Sample period:", crsp[DATE_COL].min(), "to", crsp[DATE_COL].max())
    print("Number of firm characteristics:", len(FIRM_COLS))

if __name__ == "__main__":
    main()