from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class UniformPrior:
    lower: float
    upper: float

    def cdf(self, value: np.ndarray) -> np.ndarray:
        return (value - self.lower) / (self.upper - self.lower)

    def inverse_cdf(self, quantile: np.ndarray) -> np.ndarray:
        return self.lower + quantile * (self.upper - self.lower)

    def logpdf(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        width = self.upper - self.lower
        result = np.full_like(value, -np.inf, dtype=float)
        mask = np.logical_and(value >= self.lower, value <= self.upper)
        result[mask] = -math.log(width)
        return result


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    sigma: float

    def cdf(self, value: np.ndarray) -> np.ndarray:
        z = (np.asarray(value, dtype=float) - self.mean) / self.sigma
        return np.vectorize(_standard_normal_cdf)(z)

    def inverse_cdf(self, quantile: np.ndarray) -> np.ndarray:
        quantile = np.asarray(quantile, dtype=float)
        z = np.array([_standard_normal_ppf(value) for value in quantile])
        return self.mean + self.sigma * z

    def logpdf(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        z = (value - self.mean) / self.sigma
        return -math.log(self.sigma) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z**2


@dataclass(frozen=True)
class TruncatedNormalPrior:
    mean: float
    sigma: float
    lower: float
    upper: float = math.inf

    def cdf(self, value: np.ndarray) -> np.ndarray:
        lower_cdf = _standard_normal_cdf((self.lower - self.mean) / self.sigma)
        upper_cdf = _standard_normal_cdf((self.upper - self.mean) / self.sigma)
        z = (np.asarray(value, dtype=float) - self.mean) / self.sigma
        raw_cdf = np.vectorize(_standard_normal_cdf)(z)
        return (raw_cdf - lower_cdf) / (upper_cdf - lower_cdf)

    def inverse_cdf(self, quantile: np.ndarray) -> np.ndarray:
        lower_cdf = _standard_normal_cdf((self.lower - self.mean) / self.sigma)
        upper_cdf = _standard_normal_cdf((self.upper - self.mean) / self.sigma)
        mapped = lower_cdf + np.asarray(quantile, dtype=float) * (upper_cdf - lower_cdf)
        z = np.array([_standard_normal_ppf(value) for value in mapped])
        return self.mean + self.sigma * z

    def logpdf(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        lower_cdf = _standard_normal_cdf((self.lower - self.mean) / self.sigma)
        upper_cdf = _standard_normal_cdf((self.upper - self.mean) / self.sigma)
        norm = upper_cdf - lower_cdf
        result = np.full_like(value, -np.inf, dtype=float)
        mask = np.logical_and(value >= self.lower, value <= self.upper)
        safe_value = value[mask]
        if safe_value.size > 0:
            z = (safe_value - self.mean) / self.sigma
            result[mask] = (
                -math.log(self.sigma)
                - 0.5 * math.log(2.0 * math.pi)
                - 0.5 * z**2
                - math.log(norm)
            )
        return result


@dataclass(frozen=True)
class TruncatedLogNormalPrior:
    lower: float
    upper: float
    x0: float
    sigma: float

    def cdf(self, value: np.ndarray) -> np.ndarray:
        mu = math.log(self.x0)
        lower_cdf = _standard_normal_cdf((math.log(self.lower) - mu) / self.sigma)
        upper_cdf = _standard_normal_cdf((math.log(self.upper) - mu) / self.sigma)
        z = (np.log(value) - mu) / self.sigma
        raw_cdf = np.vectorize(_standard_normal_cdf)(z)
        return (raw_cdf - lower_cdf) / (upper_cdf - lower_cdf)

    def inverse_cdf(self, quantile: np.ndarray) -> np.ndarray:
        mu = math.log(self.x0)
        lower_cdf = _standard_normal_cdf((math.log(self.lower) - mu) / self.sigma)
        upper_cdf = _standard_normal_cdf((math.log(self.upper) - mu) / self.sigma)
        mapped = lower_cdf + quantile * (upper_cdf - lower_cdf)
        z = np.array([_standard_normal_ppf(value) for value in mapped])
        return np.exp(mu + self.sigma * z)

    def logpdf(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        mu = math.log(self.x0)
        lower_cdf = _standard_normal_cdf((math.log(self.lower) - mu) / self.sigma)
        upper_cdf = _standard_normal_cdf((math.log(self.upper) - mu) / self.sigma)
        norm = upper_cdf - lower_cdf
        result = np.full_like(value, -np.inf, dtype=float)
        mask = np.logical_and(value >= self.lower, value <= self.upper)
        safe_value = value[mask]
        if safe_value.size > 0:
            z = (np.log(safe_value) - mu) / self.sigma
            result[mask] = (
                -np.log(safe_value)
                - math.log(self.sigma)
                - 0.5 * math.log(2.0 * math.pi)
                - 0.5 * z**2
                - math.log(norm)
            )
        return result


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    short_name: str
    prior: UniformPrior | NormalPrior | TruncatedNormalPrior | TruncatedLogNormalPrior


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _standard_normal_ppf(probability: float) -> float:
    # Peter John Acklam's rational approximation.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if probability <= 0.0:
        return -math.inf
    if probability >= 1.0:
        return math.inf

    if probability < plow:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def latin_hypercube_quantiles(n_eval: int, ndim: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=ndim, seed=seed)
    return sampler.random(n=n_eval)


def sample_parameter_space(
    parameter_specs: list[ParameterSpec],
    n_eval: int,
    seed: int,
) -> np.ndarray:
    quantiles = latin_hypercube_quantiles(n_eval=n_eval, ndim=len(parameter_specs), seed=seed)
    columns = []
    for index, spec in enumerate(parameter_specs):
        columns.append(spec.prior.inverse_cdf(quantiles[:, index]))
    return np.column_stack(columns)


def transform_to_prior_quantiles(
    parameter_specs: list[ParameterSpec],
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array with shape (n_samples, n_parameters)")
    if values.shape[1] != len(parameter_specs):
        raise ValueError("values column count must match the number of parameter specs")

    columns = []
    for index, spec in enumerate(parameter_specs):
        quantiles = np.asarray(spec.prior.cdf(values[:, index]), dtype=float)
        quantiles = np.clip(quantiles, 1.0e-6, 1.0 - 1.0e-6)
        columns.append(quantiles)
    return np.column_stack(columns)


def transform_from_prior_quantiles(
    parameter_specs: list[ParameterSpec],
    quantiles: np.ndarray,
) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2:
        raise ValueError("quantiles must be a 2D array with shape (n_samples, n_parameters)")
    if quantiles.shape[1] != len(parameter_specs):
        raise ValueError("quantiles column count must match the number of parameter specs")

    columns = []
    for index, spec in enumerate(parameter_specs):
        clipped = np.clip(quantiles[:, index], 1.0e-6, 1.0 - 1.0e-6)
        columns.append(np.asarray(spec.prior.inverse_cdf(clipped), dtype=float))
    return np.column_stack(columns)


def log_prior_density(
    parameter_specs: list[ParameterSpec],
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array with shape (n_samples, n_parameters)")
    if values.shape[1] != len(parameter_specs):
        raise ValueError("values column count must match the number of parameter specs")

    total = np.zeros(values.shape[0], dtype=float)
    for index, spec in enumerate(parameter_specs):
        total += np.asarray(spec.prior.logpdf(values[:, index]), dtype=float)
    return total
