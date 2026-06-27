# -*- coding: utf-8 -*-
"""
Random forest model for the rolling prediction script.

The public entry point matches model_enet.py:
train_and_select_model(X_train, y_train, X_val, y_val)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config_global import RANDOM_STATE


RF_PARAM_GRID = [
    {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 50},
    {"n_estimators": 200, "max_features": 0.33, "min_samples_leaf": 50},
    {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 100},
]


def mse(y, yhat):
    """Mean squared error used for validation."""
    return float(np.mean((y - yhat) ** 2))


def make_model(params):
    """Build a reproducible random forest regressor from a parameter dict."""
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
    )


def train_and_select_model(X_train, y_train, X_val, y_val):
    """
    Select random forest hyperparameters using the validation sample.

    The final model is refitted on train + validation data using the selected
    parameters. Returns (model, selected_params) for rolling_main.py logging.
    """
    best_mse = np.inf
    best_params = None

    for params in RF_PARAM_GRID:
        model = make_model(params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mse(y_val, pred)

        if score < best_mse:
            best_mse = score
            best_params = dict(params)

    print(f"  Random Forest best params = {best_params}, validation MSE = {best_mse:.6f}")

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    final_model = make_model(best_params)
    final_model.fit(X_tv, y_tv)

    selected_params = dict(best_params)
    selected_params["validation_mse"] = best_mse

    return final_model, selected_params
