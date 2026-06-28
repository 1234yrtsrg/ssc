# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:30:57 2026

@author: HUAWEI
"""

# -*- coding: utf-8 -*-

"""

Main rolling prediction script for the empirical asset pricing ML project.

Refactored for Person C's Ablation Experiments (消融实验专用版).

支持自由切换：

- 缩尾后数据(Winsorize-only) vs 降维Rank后数据

- 全量特征名单 vs 树模型非线性特征子名单

- 强行锁死并复用Baseline的最优超参数 (严格控制变量)

"""

import importlib

import json

import os

from pathlib import Path

import numpy as np

import pandas as pd

from config_global import *

# ============================================================

# 【核心模型选择】

# ============================================================

# 当前消融实验只跑随机森林。MODEL 是实验标签，用于读取 baseline 日志和保存文件名。
MODEL = "rf"

# RF_BACKEND 控制实际训练实现：
#   gpu: RAPIDS cuML RandomForestRegressor，适合 Linux + V100/CUDA
#   cpu: sklearn RandomForestRegressor
RF_BACKEND = os.environ.get("SSC_RF_BACKEND", "gpu").strip().lower()
if RF_BACKEND not in ["gpu", "cpu"]:
    raise ValueError("SSC_RF_BACKEND must be either 'gpu' or 'cpu'.")
MODEL_IMPL = "rf_gpu" if RF_BACKEND == "gpu" else "rf"

# ============================================================

# 🎛️ 人员C：消融实验核心配置开关 (Ablation Configuration)

# ============================================================

# ABLATION_VERSION 说明：

# 0: 基准对照组 (全量特征 + 全部Rank后数据 + 正常自动调参)

# 1: 实验组1 (名单A的10个特征不做Rank保持原始Winsorize值 + 其余55个特征保持Rank + 锁定基准超参数)

# 2: 实验组2 (B独有特征子名单不做Rank保持原始Winsorize值 + 其余特征保持Rank + 锁定基准超参数)

ABLATION_VERSION = int(os.environ.get("SSC_ABLATION_VERSION", "1"))

# 只做Winsorize、不做Rank的对比数据集路径 (人员A交出来的 parquet 文件)

WINSORIZE_DATA_PATH = Path(
    os.environ.get("SSC_WINSORIZE_DATA_PATH", PROJECT_ROOT / "panel_winsorize_only.parquet")
)

# 【核心输入1】人员B交出的"B独有特征子名单"

B_EXCLUSIVE_FEATURES = ["ocf_at", "be_me", "netis_at", "eqnetis_at", "niq_at", "age", "coskew_21d"]

# 【核心输入2】请在这里替换成人员B给你的"名单A (Top 10)"特征名字（目前先放占位符）

A_TOP10_FEATURES = ["ocfq_saleq_std", "ret_12_1", "f_score", "capex_abn", "cop_at",
                     "z_score", "ebitda_mev", "ret_60_12", "taccruals_at", "aliq_at"]

# 消融实验输出目录。基准日志仍从 LOG_OUTPUT_DIR 读取，便于复用主实验结果；
# 消融脚本自己的预测和日志统一写入这个独立文件夹。
ABLATION_OUTPUT_DIR = Path(os.environ.get("SSC_ABLATION_OUTPUT_DIR", PROJECT_ROOT / "result_ablation_rf"))
ABLATION_PRED_OUTPUT_DIR = ABLATION_OUTPUT_DIR / "preds"
ABLATION_LOG_OUTPUT_DIR = ABLATION_OUTPUT_DIR / "logs"

# 基准组（版本0）自动保存的超参数日志路径，用于读取并强行锁定超参数。
# 默认优先读主实验 logs/，如果不存在，再读本脚本独立输出目录下的 v0 日志。
BASELINE_LOG_CANDIDATES = [
    Path(os.environ["SSC_BASELINE_LOG_PATH"]) if os.environ.get("SSC_BASELINE_LOG_PATH") else None,
    LOG_OUTPUT_DIR / f"training_log_{MODEL}_yearly.csv",
    ABLATION_LOG_OUTPUT_DIR / f"training_log_{MODEL}_yearly.csv",
]

# ============================================================

MODEL_FILE_MAP = {

    "ols": "model_ols",

    "ridge": "model_ridge",

    "lasso": "model_lasso",

    "enet": "model_enet",

    "xgb": "model_xgb",

    "xgboost": "model_xgb",

    "mlp": "model_mlp",

    "nn": "model_mlp",

    "rf": "model_rf",

    "rf_gpu": "model_rf_gpu",

}

def load_model_function(model_name):

    """Load the selected model function from its model-specific script."""

    if model_name not in MODEL_FILE_MAP:

        raise ValueError(f"Unknown MODEL='{model_name}'. Valid options are {list(MODEL_FILE_MAP)}")

    module_name = MODEL_FILE_MAP[model_name]

    module = importlib.import_module(module_name)

    return module.train_and_select_model

def unpack_model_result(result):

    if isinstance(result, tuple) and len(result) == 2:

        return result[0], result[1]

    return result, {}

def to_numpy(values):

    """Convert sklearn/cuML/CuPy predictions to a one-dimensional NumPy array."""

    if hasattr(values, "get"):

        values = values.get()

    elif hasattr(values, "to_numpy"):

        values = values.to_numpy()

    return np.asarray(values).reshape(-1)

def get_train_start(test_year):

    """Return the first training year for the configured rolling design."""

    if TRAIN_WINDOW_TYPE == "expanding":

        return START_YEAR

    if TRAIN_WINDOW_TYPE == "rolling":

        return test_year - VAL_YEARS - TRAIN_YEARS

    raise ValueError("TRAIN_WINDOW_TYPE must be either 'expanding' or 'rolling'.")

def clean_rf_params(selected_params):

    """Keep only parameters understood by the RF model builders."""

    return {

        "n_estimators": int(selected_params.get("n_estimators", 200)),

        "max_features": selected_params.get("max_features", "sqrt"),

        "min_samples_leaf": int(selected_params.get("min_samples_leaf", 50)),

    }

def fit_locked_rf(X_train, y_train, X_val, y_val, selected_params):

    """Fit RF with baseline-selected params, using CPU or GPU backend."""

    X_tv = np.vstack([X_train, X_val]).astype(np.float32, copy=False)

    y_tv = np.concatenate([y_train, y_val]).astype(np.float32, copy=False)

    rf_params = clean_rf_params(selected_params)

    if MODEL_IMPL == "rf_gpu":

        from model_rf_gpu import make_model

    else:

        from model_rf import make_model

    model = make_model(rf_params)

    model.fit(X_tv, y_tv)

    locked_params = dict(rf_params)

    locked_params["backend"] = MODEL_IMPL

    if "validation_mse" in selected_params:

        locked_params["baseline_validation_mse"] = selected_params["validation_mse"]

    return model, locked_params

def get_baseline_log_path():

    """Find the RF baseline log used to lock ablation hyperparameters."""

    for path in BASELINE_LOG_CANDIDATES:

        if path is not None and path.exists():

            return path

    searched = [str(path) for path in BASELINE_LOG_CANDIDATES if path is not None]

    raise FileNotFoundError(

        "未找到基准RF日志。请先运行 ABLATION_VERSION=0 生成基准日志，或用 "

        "SSC_BASELINE_LOG_PATH 指定 training_log_rf_yearly.csv。已查找: "

        + "; ".join(searched)

    )

def print_missing_summary(df, cols, title, top_n=12):

    """Print top missing-count columns for quick data diagnosis."""

    missing = df[cols].isna().sum().sort_values(ascending=False)

    missing = missing[missing > 0]

    if missing.empty:

        return

    print(title)

    n_rows = len(df)

    for col, count in missing.head(top_n).items():

        pct = 100.0 * float(count) / float(n_rows)

        print(f"  {col}: {int(count)} ({pct:.2f}%)")

    if len(missing) > top_n:

        print(f"  ... 另有 {len(missing) - top_n} 个特征存在缺失")

def main():

    print(f"====================================================")

    print(f"运行模式: ABLATION_VERSION = {ABLATION_VERSION}")

    print(f"目标模型: {MODEL}")

    print(f"训练后端: {MODEL_IMPL}")

    print(f"====================================================")

    if MODEL != "rf":

        raise ValueError("rolling_main_ablation.py 当前按需求只支持随机森林 rf。")

    model_fn = load_model_function(MODEL_IMPL)

    # --------------------------------------------------------

    # 【开关A】完美的"控制变量数据集列替换"逻辑

    # --------------------------------------------------------

    # 无论跑哪个版本，首先加载标准的、全量做过 Rank 的清洗面板数据作为大底座

    print("【基础数据源】读取标准清洗面板数据 (Rank后):", DATA_CLEAN_PATH)

    df = pd.read_parquet(DATA_CLEAN_PATH)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["year"] = df[DATE_COL].dt.year

    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    # 如果是实验组 1 或 2，通过联合主键精准对齐，把指定列替换为"未Rank的原始Winsorize值"

    if ABLATION_VERSION in [1, 2]:

        print("【消融替换机制】正在读取未Rank的原始数据以进行列替换:", WINSORIZE_DATA_PATH)

        if not WINSORIZE_DATA_PATH.exists():

            raise FileNotFoundError(f"找不到仅Winsorize的数据文件，请确保路径正确: {WINSORIZE_DATA_PATH}")

        df_win = pd.read_parquet(WINSORIZE_DATA_PATH)

        df_win[DATE_COL] = pd.to_datetime(df_win[DATE_COL])

        # ✅ Bug1修复：对齐年份范围，防止join时产生静默NaN后被fillna(0)掩盖

        df_win["year"] = df_win[DATE_COL].dt.year

        df_win = df_win[(df_win["year"] >= START_YEAR) & (df_win["year"] <= END_YEAR)]

        # 确定当前版本需要还原成原始值的特征名单

        features_to_replace = list(A_TOP10_FEATURES) if ABLATION_VERSION == 1 else list(B_EXCLUSIVE_FEATURES)

        missing_win_cols = [c for c in features_to_replace if c not in df_win.columns]

        if missing_win_cols:

            raise ValueError(f"原始Winsorize数据表中缺失以下要替换的特征列: {missing_win_cols}")

        print(f" 正在将以下 {len(features_to_replace)} 个特征还原为原始值 (其余特征保持Rank状态): {features_to_replace}")

        # 使用 [ID, DATE] 作为联合索引进行安全替换，防止行顺序不一致导致数据错位

        df = df.set_index([ID_COL, DATE_COL])

        df_win_sub = df_win.set_index([ID_COL, DATE_COL])[features_to_replace]

        duplicated_keys = int(df_win_sub.index.duplicated().sum())

        if duplicated_keys > 0:

            raise ValueError(

                f"winsorize数据存在 {duplicated_keys} 个重复的股票ID-日期键，"

                f"请先检查 panel_winsorize_only.parquet。"

            )

        missing_keys = df.index.difference(df_win_sub.index)

        if len(missing_keys) > 0:

            raise ValueError(

                f"winsorize数据比rank数据少 {len(missing_keys)} 行股票ID-日期记录，"

                f"请检查两份数据源是否来自同一份原始样本。"

            )

        # 核心替换：把特定列覆盖成原始未Rank的值

        df[features_to_replace] = df_win_sub

        # rank前的原始特征允许存在NaN；先按月中位数补缺。
        # 这里仅提示数量，不再把原始缺失误判为行对齐失败。

        nan_after = df[features_to_replace].isna().sum().sum()

        if nan_after > 0:

            print(f" 警告：替换后的原始Winsorize特征中有 {nan_after} 个NaN，将先按月中位数补缺。")

            print_missing_summary(df, features_to_replace, " 原始Winsorize替换列缺失最多的特征:")

            print(" 正在对原始Winsorize替换列按月中位数补缺；若某月整列缺失，后续再统一填0.0。")

            df[features_to_replace] = df.groupby(DATE_COL)[features_to_replace].transform(

                lambda x: x.fillna(x.median())

            )

        df = df.reset_index()

        print(" -> 特征交叉拼接完成！已成功构建混合消融数据集。")

    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # --------------------------------------------------------

    # 【开关B】无论哪种消融版本，模型输入都必须是全量 65 个特征！

    # --------------------------------------------------------

    print("【特征集】根据消融实验控制变量法，统一使用全量特征 (All Predictors)...")

    firmcols = list(FIRM_COLS)

    missing_firmcols = [c for c in firmcols if c not in df.columns]

    if missing_firmcols:

        raise ValueError(f"面板数据中缺失以下特征列: {missing_firmcols}")

    print(f"当前实验模型实际输入的特征总数: {len(firmcols)}")

    # 缺失值最后防线处理

    total_na_before = int(df[firmcols].isna().sum().sum())

    if total_na_before > 0:

        print(f"警告：检测到缺失值 {total_na_before} 个，进行安全填充(0.0)")

        print_missing_summary(df, firmcols, "全量特征缺失最多的列:")

        df[firmcols] = df[firmcols].fillna(0.0)

    df = df.dropna(subset=[RET_COL]).copy()

    first_test_year = START_YEAR + TRAIN_YEARS + VAL_YEARS

    if first_test_year > END_YEAR:

        raise ValueError("样本总长度不足以支持设定的训练窗口与验证窗口。")

    pred_list = []

    log_list = []

    # 如果是消融组，提前把基准超参数日志加载进来

    baseline_log_df = None

    if ABLATION_VERSION in [1, 2]:

        baseline_log_path = get_baseline_log_path()

        print(f"正在载入基准超参数日志用于参数锁死: {baseline_log_path}")

        baseline_log_df = pd.read_csv(baseline_log_path)

    # 滚动窗口预测循环

    for test_year in range(first_test_year, END_YEAR + 1):

        train_start = get_train_start(test_year)

        train_end = test_year - VAL_YEARS - 1

        val_start = train_end + 1

        val_end = test_year - 1

        print(f"\n--- 测试年份 {test_year} ---")

        df_train = df[(df["year"] >= train_start) & (df["year"] <= train_end)]

        df_val = df[(df["year"] >= val_start) & (df["year"] <= val_end)]

        df_test = df[df["year"] == test_year]

        if df_train.empty or df_val.empty or df_test.empty:

            print(" -> 某数据分片为空，跳过该年份。")

            continue

        X_train = df_train[firmcols].to_numpy(dtype=np.float32)

        y_train = df_train[RET_COL].to_numpy(dtype=np.float32)

        X_val = df_val[firmcols].to_numpy(dtype=np.float32)

        y_val = df_val[RET_COL].to_numpy(dtype=np.float32)

        # --------------------------------------------------------

        # 【开关C】超参数锁定拦截逻辑

        # --------------------------------------------------------

        if ABLATION_VERSION in [1, 2]:

            # 锁定控制变量：提取当年版本0选出来的参数，强行实例化模型，绕过内部网格搜索

            year_row = baseline_log_df[

                (baseline_log_df["model"] == MODEL) & (baseline_log_df["test_year"] == test_year)

            ]

            if year_row.empty:

                raise ValueError(f"基准日志中缺失模型 {MODEL} 在 {test_year} 年的超参数记录！")

            selected_params = json.loads(year_row["selected_params"].values[0])

            print(f" [锁定参数拦截成功] {test_year}年复用基准超参数: {selected_params}")

            model, selected_params = fit_locked_rf(X_train, y_train, X_val, y_val, selected_params)

        else:

            # ABLATION_VERSION = 0 基准组，走原本的自动网格搜索调参流程

            result = model_fn(X_train, y_train, X_val, y_val)

            model, selected_params = unpack_model_result(result)

        # 预测并搜集结果

        X_test = df_test[firmcols].to_numpy(dtype=np.float32)

        df_test = df_test.copy()

        df_test["pred"] = to_numpy(model.predict(X_test))

        df_test["test_year"] = test_year

        # 标记当前预测来自哪个消融版本，方便后续读取合并

        df_test["model"] = f"{MODEL}_v{ABLATION_VERSION}" if ABLATION_VERSION > 0 else MODEL

        pred_cols = [DATE_COL, ID_COL, RET_COL, LME_COL, SIZE_GRP_COL, "pred", "test_year", "model"]

        pred_list.append(df_test[pred_cols])

        log_row = {

            "model": MODEL,

            "test_year": test_year,

            "train_start": train_start,

            "train_end": train_end,

            "val_start": val_start,

            "val_end": val_end,

            "n_train": len(df_train),

            "n_val": len(df_val),

            "n_test": len(df_test),

            "selected_params": json.dumps(selected_params),

            "ablation_version": ABLATION_VERSION

        }

        log_list.append(log_row)

    if not pred_list:

        raise RuntimeError("未生成任何预测结果。请检查年份区间与窗口配置。")

    preds = pd.concat(pred_list, axis=0).reset_index(drop=True)

    logs = pd.DataFrame(log_list)

    ABLATION_PRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ABLATION_LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 【防覆盖保护机制】文件名自动加上消融实验版本号，严防死守误覆盖基准数据

    suffix = f"_v{ABLATION_VERSION}" if ABLATION_VERSION > 0 else ""

    pred_file = ABLATION_PRED_OUTPUT_DIR / f"pred_{MODEL}_yearly{suffix}.parquet"

    log_file = ABLATION_LOG_OUTPUT_DIR / f"training_log_{MODEL}_yearly{suffix}.csv"

    preds.to_parquet(pred_file)

    logs.to_csv(log_file, index=False)

    print(f"\n====================================================")

    print("消融实验跑完啦！")

    print("预测结果成功保存至:", pred_file)

    print("实验运行日志保存至:", log_file)

    print(f"====================================================")

if __name__ == "__main__":

    main()
