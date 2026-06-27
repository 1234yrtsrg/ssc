# -*- coding: utf-8 -*-
"""
Global configuration for the empirical asset pricing ML project.

This file stores shared settings used by data cleaning, rolling prediction,
and model evaluation scripts. Keeping these settings in one place makes the
project easier to reproduce and easier to teach.
"""

from pathlib import Path

# ============================================================
# 1. Project paths
# ============================================================
# PROJECT_ROOT points to this project folder.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW_DIR = PROJECT_ROOT
DATA_PROCESSED_DIR = PROJECT_ROOT
PRED_OUTPUT_DIR = PROJECT_ROOT / "preds"
LOG_OUTPUT_DIR = PROJECT_ROOT / "logs"

RAW_DATA_PATH = PROJECT_ROOT / "us.parquet"
DATA_CLEAN_PATH = PROJECT_ROOT / "us_panel_cleaned.parquet"

# ============================================================
# 2. Column names
# ============================================================
ID_COL = "permno"
DATE_COL = "date"
RET_COL = "ret_exc_lead1m"       # next-month return used as the prediction target
PRC_COL = "prc"
EXCHCD_COL = "crsp_exchcd"
SHRCD_COL = "crsp_shrcd"
ME_COL = "me"
LME_COL = "lme"                 # market equity used for portfolio weighting
SIC_COL = "ff49"
SIZE_GRP_COL = "size_grp"

# Columns that should not be used as predictors.
# These variables are identifiers, returns, portfolio weights, or grouping variables.
OTHER_EXCLUDE = [
    SIZE_GRP_COL,
    SIC_COL,
    PRC_COL,
    EXCHCD_COL,
    SHRCD_COL,
    LME_COL,
]

# ============================================================
# 3. Firm characteristics
# ============================================================
# These are the firm characteristics used as predictors.
# They are ranked cross-sectionally in data_process.py.
FIRM_COLS = [
    'aliq_at', 'aliq_mat', 'ami_126d', 'at_be', 'at_gr1', 'at_me',
    'at_turnover', 'be_gr1a', 'be_me', 'beta_60m',
    'capex_abn', 'capx_gr1', 'cash_at', 'chcsho_12m', 'coa_gr1a', 'cop_at',
    'coskew_21d', 'cowc_gr1a', 'debt_me', 'dolvol_var_126d', 'dsale_dsga',
    'ebit_sale', 'ebitda_mev', 'eqnetis_at', 'eqnpo_12m', 'fcf_me',
    'inv_gr1', 'market_equity', 'ncoa_gr1a', 'netdebt_me', 'netis_at',
    'ni_ar1', 'niq_be', 'nncoa_gr1a', 'noa_gr1a', 'o_score',
    'oaccruals_ni', 'ocf_at', 'ocfq_saleq_std', 'ppeinv_gr1a',
    'ocf_at_chg1', 'sale_gr1', 'sale_me', 'taccruals_at',
    'turnover_126d', 'turnover_var_126d', 'z_score', 'zero_trades_21d',
    'emp_gr1', 'eqnpo_me', 'sale_emp_gr1', 'rd_sale', 'bidaskhl_21d',
    'age', 'f_score', 'gp_at', 'ivol_capm_252d', 'niq_at', 'op_at',
    'qmj', 'ret_1_0', 'ret_12_1', 'ret_60_12', 'ret_12_7', 'ret_6_1'
]

# ============================================================
# 4. Rolling-window design
# ============================================================
START_YEAR = 1965
END_YEAR = 2023

# Current design: expanding training sample + fixed-length validation sample.
TRAIN_YEARS = 20
VAL_YEARS = 10

# The code is written yearly for the U.S. sample. For shorter samples such as
# China, it can be extended later to monthly refitting.
REFIT_FREQ = "year"
#expanding or rolling
TRAIN_WINDOW_TYPE = "rolling"

# ============================================================
# 5. Reproducibility
# ============================================================
RANDOM_STATE = 42
