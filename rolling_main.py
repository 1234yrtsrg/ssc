# -*- coding: utf-8 -*-
"""
Main rolling prediction script for the empirical asset pricing ML project.

This script controls the forecasting experiment. It loads the cleaned panel,
creates train/validation/test splits, calls the selected model, and saves
out-of-sample predictions.
"""

import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from config_global import *

# ============================================================
# Models to run
# ============================================================
# Default: run both models in one command.
# Optional override without editing this file:
#   PowerShell: $env:SSC_MODELS="enet,rf"
#   CMD:        set SSC_MODELS=enet,rf
#   Linux:      SSC_MODELS=enet,rf_gpu python rolling_main.py
DEFAULT_MODELS = ["enet", "rf"]

# Map model names to file names. Each model file must define:
# train_and_select_model(X_train, y_train, X_val, y_val)
MODEL_FILE_MAP = {
    "enet": "model_enet",
    "rf": "model_rf",
    "rf_gpu": "model_rf_gpu",
}


def get_models_to_run():
    """Read the requested model list, defaulting to enet and rf."""
    raw_models = os.environ.get("SSC_MODELS")
    if not raw_models:
        return list(DEFAULT_MODELS)

    if raw_models.strip().lower() == "all":
        return list(DEFAULT_MODELS)

    models = [m.strip().lower() for m in raw_models.split(",") if m.strip()]
    unknown_models = [m for m in models if m not in MODEL_FILE_MAP]
    if unknown_models:
        raise ValueError(
            f"Unknown models in SSC_MODELS={unknown_models}. "
            f"Valid options are {list(MODEL_FILE_MAP)} or 'all'."
        )
    return models


def load_model_function(model_name):
    """Load the selected model function from its model-specific script."""
    if model_name not in MODEL_FILE_MAP:
        raise ValueError(f"Unknown MODEL='{model_name}'. Valid options are {list(MODEL_FILE_MAP)}")

    module_name = MODEL_FILE_MAP[model_name]
    module = importlib.import_module(module_name)
    return module.train_and_select_model


def unpack_model_result(result):
    """
    Support both old and new model scripts.

    New model scripts return (model, selected_params). Older scripts may return
    only model. This helper keeps rolling_main.py backward compatible.
    """
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def get_train_start(test_year):
    """Return the first training year for the configured rolling design."""
    if TRAIN_WINDOW_TYPE == "expanding":
        return START_YEAR
    if TRAIN_WINDOW_TYPE == "rolling":
        return test_year - VAL_YEARS - TRAIN_YEARS
    raise ValueError("TRAIN_WINDOW_TYPE must be either 'expanding' or 'rolling'.")


def to_numpy(values):
    """Convert model predictions to a one-dimensional NumPy array."""
    if hasattr(values, "get"):
        values = values.get()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values).reshape(-1)


def load_clean_panel():
    """Load and prepare the panel once, then reuse it for every model."""
    print("Loading cleaned panel:", DATA_CLEAN_PATH)
    df = pd.read_parquet(DATA_CLEAN_PATH)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["year"] = df[DATE_COL].dt.year

    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # Use the firm-characteristic list from config_global.py.
    # This is safer than automatically using every numeric column.
    missing_firmcols = [c for c in FIRM_COLS if c not in df.columns]
    if missing_firmcols:
        raise ValueError(f"These firm characteristics are missing from the cleaned panel: {missing_firmcols}")
    firmcols = list(FIRM_COLS)

    print(f"Using {len(firmcols)} firm characteristics as predictors.")

    # Final safety check. Ideally, missing values should already be handled in data_process.py.
    total_na_before = int(df[firmcols].isna().sum().sum())
    print(f"Total missing characteristic values before final safety fill: {total_na_before}")
    if total_na_before > 0:
        df[firmcols] = df[firmcols].fillna(0.0)

    df = df.dropna(subset=[RET_COL]).copy()

    first_test_year = START_YEAR + TRAIN_YEARS + VAL_YEARS
    if first_test_year > END_YEAR:
        raise ValueError("The sample is too short for the chosen TRAIN_YEARS and VAL_YEARS.")

    print(f"First test year = {first_test_year}")

    return df, firmcols, first_test_year


def run_one_model(model_name, df, firmcols, first_test_year):
    """Run the full rolling experiment for one model."""
    print("\n" + "=" * 70)
    print("Selected model:", model_name)
    print("=" * 70)

    model_fn = load_model_function(model_name)

    PRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    YEARLY_OUTPUT_DIR = PRED_OUTPUT_DIR / f"pred_{model_name}_by_year"
    YEARLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_log_file = LOG_OUTPUT_DIR / f"training_log_{model_name}_partial.csv"

    pred_list = []
    log_list = []

    for test_year in range(first_test_year, END_YEAR + 1):
        train_start = get_train_start(test_year)
        train_end = test_year - VAL_YEARS - 1
        val_start = train_end + 1
        val_end = test_year - 1

        print(f"\n=== Test year {test_year} ===")
        print(f"  Train: {train_start}-{train_end}")
        print(f"  Val:   {val_start}-{val_end}")
        print(f"  Test:  {test_year}")

        df_train = df[(df["year"] >= train_start) & (df["year"] <= train_end)]
        df_val = df[(df["year"] >= val_start) & (df["year"] <= val_end)]
        df_test = df[df["year"] == test_year]

        if df_train.empty or df_val.empty or df_test.empty:
            print("  -> Skip because one split is empty.")
            continue

        # Use float32 to reduce memory usage. This is usually enough for ML prediction.
        X_train = df_train[firmcols].to_numpy(dtype=np.float32)
        y_train = df_train[RET_COL].to_numpy(dtype=np.float32)
        X_val = df_val[firmcols].to_numpy(dtype=np.float32)
        y_val = df_val[RET_COL].to_numpy(dtype=np.float32)

        result = model_fn(X_train, y_train, X_val, y_val)
        model, selected_params = unpack_model_result(result)

        X_test = df_test[firmcols].to_numpy(dtype=np.float32)
        df_test = df_test.copy()
        df_test["pred"] = to_numpy(model.predict(X_test))
        df_test["test_year"] = test_year
        df_test["model"] = model_name

        pred_cols = [DATE_COL, ID_COL, RET_COL, LME_COL, SIZE_GRP_COL, "pred", "test_year", "model"]
        year_preds = df_test[pred_cols]
        pred_list.append(year_preds)

        log_row = {
            "model": model_name,
            "test_year": test_year,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "n_train": len(df_train),
            "n_val": len(df_val),
            "n_test": len(df_test),
            "selected_params": json.dumps(selected_params),
        }
        log_list.append(log_row)

        year_pred_file = YEARLY_OUTPUT_DIR / f"pred_{model_name}_{test_year}.parquet"
        year_preds.to_parquet(year_pred_file)
        pd.DataFrame(log_list).to_csv(partial_log_file, index=False)
        print("  Saved yearly predictions to:", year_pred_file)
        print("  Updated partial training log:", partial_log_file)

    if not pred_list:
        raise RuntimeError("No predictions were generated. Please check the sample period and window settings.")

    preds = pd.concat(pred_list, axis=0).reset_index(drop=True)
    logs = pd.DataFrame(log_list)

    pred_file = PRED_OUTPUT_DIR / f"pred_{model_name}_yearly.parquet"
    log_file = LOG_OUTPUT_DIR / f"training_log_{model_name}_yearly.csv"

    preds.to_parquet(pred_file)
    logs.to_csv(log_file, index=False)

    print("\nSaved predictions to:", pred_file)
    print("Saved training log to:", log_file)
    print("Prediction sample:")
    print(preds.head())


def main():
    models = get_models_to_run()
    print("Models to run:", models)

    df, firmcols, first_test_year = load_clean_panel()

    for model_name in models:
        run_one_model(model_name, df, firmcols, first_test_year)


if __name__ == "__main__":
    main()
