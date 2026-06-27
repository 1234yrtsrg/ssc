# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:22:25 2026

@author: HUAWEI
"""

# scripts/evaluate_model.py

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
# Regression
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
from linearmodels import FamaMacBeth, BetweenOLS
# Save data
import _pickle as pickle
from pandas.tseries.offsets import MonthEnd, MonthBegin, Week
import os
import re
import seaborn as sns; sns.set(color_codes=True)
from linearmodels.panel import PooledOLS, PanelOLS
import statsmodels.api as sm
import functools
from statsmodels.distributions.empirical_distribution import ECDF

from pathlib import Path


import os
from config_global import *



def wavg(group, avg_name, weight_name, VW):
    d = group[avg_name]
    if VW == 'vw':
        w = group[weight_name]
    else:
        w = group[weight_name] / group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

def ecdf_sort(data, rule, var):
    ecdf = ECDF(data.loc[data[rule]==1, [var]].values.flatten())
    return pd.Series(ecdf(data[var].values.flatten()), index=data.index, name='cdf')

def none_microcap_sort(data, weight, ret, prd, num_ports, port_weight):
    temp = data[['date','permno',weight,ret,prd,'size_grp']].copy()
    temp = temp.loc[(~temp[prd].isna()) & (~temp[ret].isna())]
    temp['bp_stock'] = np.where(temp['size_grp'].isin(['mega','large','small']),1,0)
    temp = temp.groupby('date').filter(lambda x: x['bp_stock'].sum()>=10)
    temp[prd] = temp.groupby('date')[prd].transform(lambda x: (2*x.rank(pct=True)-1)/2)
    temp = temp.set_index('permno')
    cdf_ = temp.groupby('date').apply(ecdf_sort, 'bp_stock', prd).reset_index()
    temp = temp.reset_index()
    temp = pd.merge(temp, cdf_, how='inner', on=['date','permno'])
    temp['cdf_rnk'] = temp.groupby('date')['cdf'].rank(ascending = 1)
    temp['cdf'] = np.where(temp['cdf_rnk']==1, 0.00000001, temp['cdf'])
    temp['pf'] = temp.groupby('date')['cdf'].transform(lambda x: np.ceil(x*num_ports))
    temp['pf'] = np.where(temp['pf']==0, 1, temp['pf'])
    port_pf = temp.groupby(['date','pf']).apply(wavg, ret, weight, port_weight).reset_index()
    port_pf.columns = ['date','pf','ret']
    port_pf = pd.pivot(port_pf,
                       index = 'date',
                       columns = 'pf',
                       values = 'ret')
    #port_pf = port_pf.reset_index()
    #port_pf.columns = ['date','low','med','high']
    port_pf['H_L'] = port_pf[num_ports] - port_pf[1]
    hedge_port = port_pf[['H_L']]
    portfolio_rnt_mean = port_pf.mean()
    H_L_t_stat = smf.ols('H_L ~ 1',data=port_pf).fit(cov_type='HAC',cov_kwds={'maxlags':5}).tvalues[0]
    portfolio_rnt_mean = pd.concat([portfolio_rnt_mean, pd.Series({'H_L_t_stat': H_L_t_stat})])
    return (hedge_port, portfolio_rnt_mean, port_pf)
# ========= 1. OOS R^2 (panel level) =========

def oos_r2_panel(df, ret_col, pred_col):
    """
    Compute panel out-of-sample R^2:
      1 - sum (r - r_hat)^2 / sum r^2
    using all (i,t) in the test sample.
    """
    residual = df[ret_col] - df[pred_col]
    num = (residual ** 2).sum()
    den = (df[ret_col] ** 2).sum()
    return 1.0 - num / den


# ========= 2. Portfolio metrics: Sharpe & Drawdown =========

def sharpe_ratio_nw(ret_series, freq=12, lags=3):
    """
    Newey–West adjusted annualized Sharpe ratio.

    ret_series: pandas Series of returns (e.g., monthly LS portfolio)
    freq:       12 for monthly data, 252 for daily, etc.
    lags:       NW lags (3 is common for monthly data)
    """
    r = ret_series.dropna().values
    T = len(r)
    if T == 0:
        return np.nan, np.nan

    # OLS of r_t on constant, with HAC (NW) covariance
    X = np.ones((T, 1))
    ols = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    mu_hat = ols.params[0]      # sample mean
    se_mu  = ols.bse[0]         # NW standard error of the mean

    t_stat = mu_hat / se_mu     # HAC t-stat of mean
    sharpe_ann = t_stat * np.sqrt(freq / T)

    return sharpe_ann, t_stat

def max_drawdown(ret_series):
    """
    MaxDD using cumulative log returns, as in the GKX definition:
      Y_t = cumulative log(1+r), MaxDD = max(Y_t1 - Y_t2).
    Returns MaxDD in log terms and the corresponding simple-return loss.
    """
    log_ret = np.log1p(ret_series)
    cum_log = log_ret.cumsum()
    running_max = cum_log.cummax()
    drawdown = running_max - cum_log

    max_dd_log = drawdown.max()
    # convert to simple-return equivalent drop
    max_dd_simple = 1.0 - np.exp(-max_dd_log)
    return float(max_dd_log), float(max_dd_simple)


# ========= 3. Turnover (requires weights) =========

def turnover_from_weights(weights_panel, ret_panel, date_col, id_col):
    """
    Compute average monthly turnover given:
      weights_panel: DataFrame with columns [date, permno, w_t]
      ret_panel:     DataFrame with columns [date, permno, r_{t+1}]
    This implements the formula:
      Turnover = (1/T) * sum_t sum_i | w_{i,t+1}
                  - w_{i,t}(1+r_{i,t+1}) / (1 + sum_j w_{j,t} r_{j,t+1}) |

    For now this is a template: you can adapt it to your portfolio helper's output.
    """
    # merge weights and next-month returns
    df = weights_panel.merge(ret_panel, on=[date_col, id_col], how="inner")

    # sort by date for group operations
    df = df.sort_values([date_col, id_col])

    # shift weights to get w_{i,t} and w_{i,t+1}
    # here we assume weights_panel is already "w_{i,t}" at each date
    # so we build w_{i,t+1} by shifting within permno
    df["w_t"]   = df["weight"]
    df["w_tp1"] = df.groupby(id_col)["w_t"].shift(-1)
    df["r_tp1"] = df["ret"]   # rename for clarity if needed

    # compute scaling term: 1 + sum_j w_{j,t} r_{j,t+1}  (per date)
    df["w_t_r_tp1"] = df["w_t"] * df["r_tp1"]
    scale = df.groupby(date_col)["w_t_r_tp1"].sum()
    scale = 1.0 + scale
    df = df.join(scale.rename("scale"), on=date_col)

    # theoretical "post-return" weights from previous period
    df["w_t_post"] = df["w_t"] * (1.0 + df["r_tp1"]) / df["scale"]

    # turnover per date: sum_i | w_{i,t+1} - w_{i,t_post} |
    df["abs_change"] = (df["w_tp1"] - df["w_t_post"]).abs()
    turnover_t = df.groupby(date_col)["abs_change"].sum()

    # average over time
    avg_turnover = turnover_t.mean()
    return float(avg_turnover)

#%% 
# ========= 4. MAIN EVALUATION PIPELINE =========
# ========= CHOOSE MODEL TO EVALUATE =========
MODEL = os.environ.get("SSC_MODEL", "enet")   # "enet", "rf", "rf_gpu"


def flatten_metrics(metrics):
    """Make the metrics dictionary easy to save as one CSV row."""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, pd.Series):
            for series_key, series_value in value.items():
                flat[f"{key}_{series_key}"] = series_value
        else:
            flat[key] = value
    return flat


def save_metrics(metrics):
    """Save the printed evaluation output into the logs folder."""
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flat_metrics = flatten_metrics(metrics)
    csv_file = LOG_OUTPUT_DIR / f"evaluation_metrics_{MODEL}.csv"
    txt_file = LOG_OUTPUT_DIR / f"evaluation_metrics_{MODEL}.txt"

    pd.DataFrame([flat_metrics]).to_csv(csv_file, index=False)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation metrics for MODEL={MODEL}\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for key, value in flat_metrics.items():
            f.write(f"{key}: {value}\n")

    print("Saved evaluation metrics to:", csv_file)
    print("Saved evaluation metrics text to:", txt_file)


def main():
    pred_dir = Path(PRED_OUTPUT_DIR)
    pred_file = pred_dir / f"pred_{MODEL}_yearly.parquet"

    print("Loading predictions from:", pred_file)
    df_pred = pd.read_parquet(pred_file)

    # ---------- OOS R^2 ----------
    r2_oos = oos_r2_panel(df_pred, RET_COL, "pred")

    # ---------- Portfolio returns (using your existing helper) ----------
    # from your_portfolio_helper import build_decile_spread
    weight = 'lme'
    num_ports = 10
    port_weight = 'vw'
    (port_all,mean_all,port_ind_all) = none_microcap_sort(df_pred, weight, RET_COL, 'pred', num_ports, port_weight)

    # -------------------------------------------------------------------------

    ret_ls = port_all["H_L"]
    sharpe_ann, t_mu = sharpe_ratio_nw(ret_ls)
    mean=mean_all
    max_dd_log, max_dd_simple = max_drawdown(ret_ls)
    
    
    metrics = {
            "model": MODEL,
            "r2_oos": r2_oos,
            "sharpe_ann": sharpe_ann,
            "t_mu": t_mu,
            "mean_ls": mean,
            "max_dd_log": max_dd_log,
            "max_dd_simple": max_dd_simple,
        }
    return metrics
    #
    # If you have weights from your helper, you can compute turnover:
    # avg_turn = turnover_from_weights(weights_panel, ret_panel, DATE_COL, ID_COL)
    # print(f"  Average monthly turnover: {avg_turn:.3f}")

    print("\n(Plug in your decile-portfolio helper to complete Sharpe/Drawdown/Turnover.)")


if __name__ == "__main__":
    results = main()
    print(results)
    save_metrics(results)
