from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut


def build_kernel(n_features: int, *, fit_white_noise: bool = False) -> ConstantKernel:
    kernel = ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.full(n_features, 0.25),
        length_scale_bounds=(1.0e-2, 10.0),
        nu=2.5,
    )
    if fit_white_noise:
        kernel += WhiteKernel(noise_level=1.0e-3, noise_level_bounds=(1.0e-8, 1.0))
    return kernel


def fit_scaled_gp(
    x: np.ndarray,
    y: np.ndarray,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool = True,
    alpha: np.ndarray | float | None = None,
    fit_white_noise: bool = False,
) -> tuple[GaussianProcessRegressor, float, float]:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std == 0.0:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    alpha_scaled = None
    if alpha is not None:
        alpha_scaled = np.asarray(alpha, dtype=float) / (y_std ** 2)
        alpha_scaled = np.maximum(alpha_scaled, 1.0e-12)
    kwargs = {
        "kernel": build_kernel(x.shape[1], fit_white_noise=fit_white_noise),
        "normalize_y": False,
        "n_restarts_optimizer": n_restarts_optimizer,
        "optimizer": "fmin_l_bfgs_b" if optimize_hyperparameters else None,
        "random_state": 42,
    }
    if alpha_scaled is not None:
        kwargs["alpha"] = alpha_scaled
    model = GaussianProcessRegressor(**kwargs)
    model.fit(x, y_scaled)
    return model, y_mean, y_std


def predict_scaled_gp(
    model: GaussianProcessRegressor,
    y_mean: float,
    y_std: float,
    x: np.ndarray,
    *,
    return_std: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if not return_std:
        pred_scaled = model.predict(x, return_std=False)
        return y_mean + y_std * pred_scaled, np.zeros_like(pred_scaled)
    pred_scaled, pred_std_scaled = model.predict(x, return_std=True)
    return y_mean + y_std * pred_scaled, y_std * pred_std_scaled

def fit_cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    n_restarts_optimizer: int,
    cv_folds: int,
    optimize_hyperparameters: bool = True,
    alpha: np.ndarray | None = None,
    fit_white_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = x.shape[0]
    splitter = LeaveOneOut() if cv_folds >= n_samples else KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42,
    )
    predictions = np.zeros(n_samples)
    std_predictions = np.zeros(n_samples)

    for train_index, test_index in splitter.split(x):
        model, y_mean, y_std = fit_scaled_gp(
            x=x[train_index],
            y=y[train_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            alpha=alpha[train_index] if alpha is not None else None,
            fit_white_noise=fit_white_noise,
        )
        pred, pred_std = predict_scaled_gp(model, y_mean, y_std, x[test_index])
        predictions[test_index] = pred
        std_predictions[test_index] = pred_std

    return predictions, std_predictions


def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    predicted_std: np.ndarray,
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    mae = float(mean_absolute_error(observed, predicted))
    r2 = float(r2_score(observed, predicted))
    coverage_1sigma = float(np.mean(np.abs(predicted - observed) <= predicted_std))
    coverage_2sigma = float(np.mean(np.abs(predicted - observed) <= 2.0 * predicted_std))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
    }
