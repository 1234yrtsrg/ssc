# -*- coding: utf-8 -*-
"""
GPU random forest model for the rolling prediction script.

This uses RAPIDS cuML when available. It keeps the same public entry point as
model_rf.py: train_and_select_model(X_train, y_train, X_val, y_val).
"""

import numpy as np

from config_global import RANDOM_STATE

try:
    from cuml.ensemble import RandomForestRegressor
except ImportError as exc:
    raise ImportError(
        "MODEL='rf_gpu' requires RAPIDS cuML, but it is not installed in this "
        "Python environment. Use MODEL='rf' for sklearn CPU random forest, or "
        "install RAPIDS/cuML in a CUDA-capable environment."
    ) from exc


RF_PARAM_GRID = [
    {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 50},
    {"n_estimators": 200, "max_features": 0.33, "min_samples_leaf": 50},
    {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 100},
]


def to_numpy(values):
    """Convert cuML/CuPy outputs back to NumPy for pandas assignment/logging."""
    if hasattr(values, "get"):
        return values.get()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def mse(y, yhat):
    """Mean squared error used for validation."""
    yhat = to_numpy(yhat).reshape(-1)
    return float(np.mean((y - yhat) ** 2))


def make_model(params):
    """Build a reproducible GPU random forest regressor."""
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
        bootstrap=True,
    )


def train_and_select_model(X_train, y_train, X_val, y_val):
    """
    Select GPU random forest hyperparameters using the validation sample.

    The final model is refitted on train + validation data using the selected
    parameters. Returns (model, selected_params) for rolling_main.py logging.
    """
    best_mse = np.inf
    best_params = None

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    for params in RF_PARAM_GRID:
        model = make_model(params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mse(y_val, pred)

        if score < best_mse:
            best_mse = score
            best_params = dict(params)

    print(f"  GPU Random Forest best params = {best_params}, validation MSE = {best_mse:.6f}")

    X_tv = np.vstack([X_train, X_val]).astype(np.float32, copy=False)
    y_tv = np.concatenate([y_train, y_val]).astype(np.float32, copy=False)

    final_model = make_model(best_params)
    final_model.fit(X_tv, y_tv)

    selected_params = dict(best_params)
    selected_params["validation_mse"] = best_mse
    selected_params["backend"] = "cuml"

    return final_model, selected_params
