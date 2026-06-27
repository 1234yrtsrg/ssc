# -*- coding: utf-8 -*-
"""
Elastic Net model for the rolling prediction script.

This implementation uses a small coordinate-descent solver built on NumPy.
It avoids sklearn's ElasticNet coordinate-descent extension, which can crash in
some local binary environments before Python has a chance to raise an error.
"""

import numpy as np


ENET_ALPHAS = np.logspace(-5, -1, 9)
ENET_L1_RATIOS = [0.1, 0.5, 0.75, 0.9]


def mse(y, yhat):
    """Mean squared error used for validation."""
    return float(np.mean((y - yhat) ** 2))


def soft_threshold(value, penalty):
    """Soft-thresholding operator for the L1 part of Elastic Net."""
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


class ElasticNetNumpy:
    """Elastic Net regressor with a sklearn-like predict method."""

    def __init__(self, alpha, l1_ratio, max_iter=5000, tol=1e-5):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        problem = prepare_problem(X, y)
        coef_scaled = solve_from_problem(
            problem,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.set_solution(problem, coef_scaled)
        return self

    def set_solution(self, problem, coef_scaled):
        """Store a solved coefficient vector from a prepared standardized problem."""
        self.x_mean_ = problem["x_mean"]
        self.x_scale_ = problem["x_scale"]
        self.y_mean_ = problem["y_mean"]
        self.coef_ = coef_scaled / self.x_scale_
        self.intercept_ = self.y_mean_ - float(np.sum(self.x_mean_ * self.coef_))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sum(X * self.coef_, axis=1) + self.intercept_


def prepare_problem(X, y):
    """Standardize data once and build the Gram matrix used by coordinate descent."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    y_mean = float(y.mean())

    Xs = (X - x_mean) / x_scale
    yc = y - y_mean
    n_obs = Xs.shape[0]
    n_features = Xs.shape[1]

    gram = np.empty((n_features, n_features), dtype=np.float64)
    for i in range(n_features):
        for j in range(i, n_features):
            value = float(np.sum(Xs[:, i] * Xs[:, j]) / n_obs)
            gram[i, j] = value
            gram[j, i] = value

    xy = np.empty(n_features, dtype=np.float64)
    for i in range(n_features):
        xy[i] = float(np.sum(Xs[:, i] * yc) / n_obs)

    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "gram": gram,
        "xy": xy,
    }


def solve_from_problem(problem, alpha, l1_ratio, max_iter=5000, tol=1e-5):
    """Solve one Elastic Net parameter pair from a prepared Gram problem."""
    gram = problem["gram"]
    xy = problem["xy"]
    coef = np.zeros(xy.size, dtype=np.float64)
    l1_penalty = float(alpha) * float(l1_ratio)
    l2_penalty = float(alpha) * (1.0 - float(l1_ratio))

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(coef.size):
            old_coef = coef[j]
            residual_correlation = xy[j] - (np.sum(gram[j] * coef) - gram[j, j] * old_coef)
            coef[j] = soft_threshold(residual_correlation, l1_penalty) / (gram[j, j] + l2_penalty)
            max_change = max(max_change, abs(coef[j] - old_coef))

        if max_change < tol:
            break

    return coef


def train_and_select_model(X_train, y_train, X_val, y_val):
    """
    Select the Elastic Net penalty parameters using the validation sample.

    The final model is refitted on train + validation data using the selected
    alpha and l1_ratio. Returns (model, selected_params) for rolling_main.py.
    """
    best_mse = np.inf
    best_alpha = None
    best_l1_ratio = None
    train_problem = prepare_problem(X_train, y_train)

    for alpha in ENET_ALPHAS:
        for l1_ratio in ENET_L1_RATIOS:
            model = ElasticNetNumpy(alpha=alpha, l1_ratio=l1_ratio)
            coef_scaled = solve_from_problem(train_problem, alpha=alpha, l1_ratio=l1_ratio)
            model.set_solution(train_problem, coef_scaled)
            score = mse(y_val, model.predict(X_val))

            if score < best_mse:
                best_mse = score
                best_alpha = float(alpha)
                best_l1_ratio = float(l1_ratio)

    print(
        f"  Elastic Net best alpha = {best_alpha:.6g}, "
        f"best l1_ratio = {best_l1_ratio}, validation MSE = {best_mse:.6f}",
        flush=True,
    )

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    final_model = ElasticNetNumpy(alpha=best_alpha, l1_ratio=best_l1_ratio)
    tv_problem = prepare_problem(X_tv, y_tv)
    coef_scaled = solve_from_problem(tv_problem, alpha=best_alpha, l1_ratio=best_l1_ratio)
    final_model.set_solution(tv_problem, coef_scaled)

    selected_params = {
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
        "validation_mse": best_mse,
    }

    return final_model, selected_params
