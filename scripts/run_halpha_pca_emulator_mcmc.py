from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    emcee = None

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    corner = None

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

from fit_halpha_allz_gp_cv import (
    INPUT_COLUMNS,
    INPUT_PARAMETER_SPECS,
    _apply_filters,
    _fit_scaled_gp,
    _limit_draws_per_eval,
    _log10_with_floor,
    _output_columns,
    _predict_scaled_gp,
)
from galacticus_emu.lhs import (
    log_prior_density,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
)
from galacticus_emu.mcmc_corner import corner_with_log10_priors
from galacticus_emu.mcmc_trace import make_trace_plot


SOBRAL_LABELS = ["Z1", "Z2", "Z3", "Z4"]
ANALYSIS_NAMES = {
    "Z1": "luminosityFunctionHalphaSobral2013HiZELSZ1",
    "Z2": "luminosityFunctionHalphaSobral2013HiZELSZ2",
    "Z3": "luminosityFunctionHalphaSobral2013HiZELSZ3",
    "Z4": "luminosityFunctionHalphaSobral2013HiZELSZ4",
}


class MeanCenterer:
    def fit(self, y: np.ndarray) -> "MeanCenterer":
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.ones(y.shape[1], dtype=float)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y - self.mean_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y + self.mean_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run emcee MCMC using PCA-decomposed Halpha LF emulators for the Sobral Z1-Z4 "
            "luminosity functions."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--sobral-labels", nargs="*", default=SOBRAL_LABELS, choices=SOBRAL_LABELS)
    parser.add_argument("--train-draws-per-eval", type=int, default=8)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="unscaled")
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--n-walkers", type=int, default=48)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--init-center", choices=["prior_center", "best_training"], default="best_training")
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mcmc-dir-name", default="emulator_halpha_pca_mcmc")
    parser.add_argument("--figures-dir-name", default="figures_halpha_pca_mcmc")
    parser.add_argument("--output-prefix", default="halpha_pca_allz")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Skip emulator fitting and MCMC; regenerate plots from existing saved outputs.",
    )
    return parser.parse_args()


def _make_preprocessor(pca_scaling: str):
    if pca_scaling == "standardized":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    if pca_scaling == "unscaled":
        return MeanCenterer()
    raise ValueError(f"Unknown pca_scaling={pca_scaling!r}")


def _load_reduced_hdf5(campaign_root: Path) -> Path:
    reduced = next((campaign_root / "evaluations").glob("*/galacticus_reduced.hdf5"), None)
    if reduced is not None:
        return reduced
    raw = next((campaign_root / "evaluations").glob("*/galacticus.hdf5"), None)
    if raw is not None:
        return raw
    raise FileNotFoundError(f"No Galacticus HDF5 found under {campaign_root / 'evaluations'}")


def _load_target_block(
    campaign_root: Path, sobral_label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hdf5_path = _load_reduced_hdf5(campaign_root)
    analysis_name = ANALYSIS_NAMES[sobral_label]
    with h5py.File(hdf5_path, "r") as handle:
        group = handle["analyses"][analysis_name]
        luminosity = np.asarray(group["luminosity"][:], dtype=float)
        target_linear = np.asarray(group["luminosityFunctionTarget"][:], dtype=float)
        covariance_linear = np.asarray(group["luminosityFunctionCovarianceTarget"][:], dtype=float)
    if np.any(target_linear <= 0.0):
        raise ValueError(f"{sobral_label} target contains non-positive LF values; cannot transform to log-space.")
    jacobian = np.diag(1.0 / (np.log(10.0) * target_linear))
    target_log10 = np.log10(target_linear)
    covariance_log10 = jacobian @ covariance_linear @ jacobian
    return luminosity, target_linear, covariance_linear, target_log10, covariance_log10


def _block_diag(arrays: list[np.ndarray]) -> np.ndarray:
    total = sum(array.shape[0] for array in arrays)
    block = np.zeros((total, total), dtype=float)
    start = 0
    for array in arrays:
        n = array.shape[0]
        block[start : start + n, start : start + n] = array
        start += n
    return block


def _best_training_theta(
    table: pd.DataFrame,
    target_by_label: dict[str, np.ndarray],
    output_columns_by_label: dict[str, list[str]],
    min_log10_lf: float,
) -> np.ndarray:
    scores = np.zeros(len(table), dtype=float)
    for sobral_label, columns in output_columns_by_label.items():
        y_linear = table[columns].to_numpy(dtype=float)
        y_log10, _, _, _ = _log10_with_floor(y_linear, min_log10_lf)
        residual = y_log10 - target_by_label[sobral_label][None, :]
        scores += np.mean(residual**2, axis=1)
    index = int(np.argmin(scores))
    return table.iloc[index][INPUT_COLUMNS].to_numpy(dtype=float)


def _fit_pca_bundle(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sobral_label: str,
    pca_components: int,
    pca_scaling: str,
    n_restarts_optimizer: int,
    random_state: int,
) -> dict[str, object]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y)
    n_components_actual = min(pca_components, y_scaled.shape[0], y_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    coefficients = pca.fit_transform(y_scaled)

    component_models: list[dict[str, object]] = []
    for component_index in range(n_components_actual):
        print(
            f"{sobral_label}: fitting PCA component {component_index + 1}/{n_components_actual}",
            flush=True,
        )
        model, y_mean, y_std = _fit_scaled_gp(
            x,
            coefficients[:, component_index],
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        component_models.append(
            {
                "model": model,
                "y_mean": y_mean,
                "y_std": y_std,
                "kernel": str(model.kernel_),
            }
        )

    return {
        "sobral_label": sobral_label,
        "scaler": scaler,
        "pca": pca,
        "component_models": component_models,
        "n_components_actual": n_components_actual,
        "pca_scaling": pca_scaling,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _predict_pca_bundle(bundle: dict[str, object], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_samples = x.shape[0]
    n_components = int(bundle["n_components_actual"])
    coefficient_predictions = np.zeros((n_samples, n_components), dtype=float)
    coefficient_stds = np.zeros_like(coefficient_predictions)
    for component_index, component_bundle in enumerate(bundle["component_models"]):
        pred, pred_std = _predict_scaled_gp(
            component_bundle["model"],
            component_bundle["y_mean"],
            component_bundle["y_std"],
            x,
        )
        coefficient_predictions[:, component_index] = pred
        coefficient_stds[:, component_index] = pred_std

    pca: PCA = bundle["pca"]
    scaler = bundle["scaler"]
    y_scaled = pca.inverse_transform(coefficient_predictions)
    y_pred = scaler.inverse_transform(y_scaled)

    coefficient_var = coefficient_stds**2
    y_var_scaled = coefficient_var @ (pca.components_**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]
    return y_pred, y_std


def _log10_normal_to_linear_moments(
    mean_log10: np.ndarray, variance_log10: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ln10 = float(np.log(10.0))
    mu_ln = ln10 * mean_log10
    sigma2_ln = (ln10**2) * variance_log10
    mean_linear = np.exp(mu_ln + 0.5 * sigma2_ln)
    variance_linear = (np.exp(sigma2_ln) - 1.0) * np.exp(2.0 * mu_ln + sigma2_ln)
    return mean_linear, variance_linear


class HalphaPCAEmulatorPosterior:
    def __init__(
        self,
        *,
        parameter_specs,
        model_bundles: dict[str, dict[str, object]],
        target_vector: np.ndarray,
        target_covariance: np.ndarray,
        output_slices: dict[str, slice],
    ):
        self.parameter_specs = parameter_specs
        self.model_bundles = model_bundles
        self.target_vector = target_vector
        self.target_covariance = target_covariance
        self.output_slices = output_slices

    @staticmethod
    def _gaussian_log_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        return -0.5 * (quadratic + logdet + len(residual) * np.log(2.0 * np.pi))

    def predict_batch_log10(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        x = transform_to_prior_quantiles(self.parameter_specs, theta)
        mean_blocks: list[np.ndarray] = []
        var_blocks: list[np.ndarray] = []
        for sobral_label in self.output_slices:
            pred, pred_std = _predict_pca_bundle(self.model_bundles[sobral_label], x)
            mean_blocks.append(pred)
            var_blocks.append(pred_std**2)
        return np.concatenate(mean_blocks, axis=1), np.concatenate(var_blocks, axis=1)

    def predict_log10(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_batch_log10(theta[None, :])
        return means[0], variances[0]

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means_log10, variances_log10 = self.predict_batch_log10(theta)
        return _log10_normal_to_linear_moments(means_log10, variances_log10)

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_batch(theta[None, :])
        return means[0], variances[0]

    def log_probability(self, theta: np.ndarray) -> float:
        log_prior = float(log_prior_density(self.parameter_specs, theta[None, :])[0])
        if not np.isfinite(log_prior):
            return -np.inf
        mean, variance = self.predict(theta)
        covariance = self.target_covariance + np.diag(variance + 1.0e-8)
        residual = self.target_vector - mean
        try:
            log_like = self._gaussian_log_likelihood(residual, covariance)
        except np.linalg.LinAlgError:
            return -np.inf
        if not np.isfinite(log_like):
            return -np.inf
        return log_prior + log_like

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        log_prior = np.asarray(log_prior_density(self.parameter_specs, theta), dtype=float)
        result = np.full(theta.shape[0], -np.inf, dtype=float)
        valid = np.isfinite(log_prior)
        if not np.any(valid):
            return result
        means, variances = self.predict_batch(theta[valid])
        residuals = self.target_vector[None, :] - means
        valid_indices = np.where(valid)[0]
        for local_index, sample_index in enumerate(valid_indices):
            covariance = self.target_covariance + np.diag(variances[local_index] + 1.0e-8)
            try:
                log_like = self._gaussian_log_likelihood(residuals[local_index], covariance)
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(log_like):
                result[sample_index] = log_prior[sample_index] + log_like
        return result


def _make_trace_plot(
    samples: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    parameter_specs=None,
    burn_in: int = 0,
) -> dict[str, float]:
    return make_trace_plot(
        samples,
        labels,
        output_path,
        parameter_specs=parameter_specs,
        burn_in=burn_in,
        row_height=2.0,
        alpha=0.25,
        linewidth=0.6,
    )


def _plot_best_fit_lfs(
    sobral_labels: list[str],
    luminosity_by_label: dict[str, np.ndarray],
    target_linear_by_label: dict[str, np.ndarray],
    target_covariance_linear_by_label: dict[str, np.ndarray],
    target_by_label: dict[str, np.ndarray],
    target_covariance_by_label: dict[str, np.ndarray],
    prediction_by_label: dict[str, np.ndarray],
    prediction_std_by_label: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    n_panels = len(sobral_labels)
    ncols = 1 if n_panels == 1 else 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    redshift_titles = {"Z1": "z = 0.40", "Z2": "z = 0.84", "Z3": "z = 1.47", "Z4": "z = 2.23"}
    for axis, sobral_label in zip(axes_flat, sobral_labels, strict=False):
        x = np.log10(luminosity_by_label[sobral_label])
        y_true_linear = target_linear_by_label[sobral_label]
        y_true_linear_std = np.sqrt(np.diag(target_covariance_linear_by_label[sobral_label]))
        y_true = target_by_label[sobral_label]
        lower_linear = np.maximum(y_true_linear - y_true_linear_std, 1.0e-30)
        upper_linear = y_true_linear + y_true_linear_std
        yerr_lower = y_true - np.log10(lower_linear)
        yerr_upper = np.log10(upper_linear) - y_true
        y_pred = prediction_by_label[sobral_label]
        y_pred_std = prediction_std_by_label[sobral_label]
        axis.errorbar(
            x,
            y_true,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            fmt="o",
            color="0.15",
            label="Sobral target",
        )
        axis.plot(x, y_pred, color="tab:blue", lw=1.8, label="PCA emulator best fit")
        axis.fill_between(x, y_pred - y_pred_std, y_pred + y_pred_std, color="tab:blue", alpha=0.2)
        axis.set_title(redshift_titles.get(sobral_label, sobral_label))
        axis.set_xlabel(r"$\log_{10}(L_{\mathrm{H}\alpha}/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi$")
        axis.set_ylim(-6.0, -1.0)
        axis.grid(alpha=0.22)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _replot_from_saved_outputs(campaign_root: Path, mcmc_root: Path, figures_root: Path, output_prefix: str) -> None:
    posterior_path = mcmc_root / f"{output_prefix}_posterior_samples.csv"
    chain_path = mcmc_root / f"{output_prefix}_chain.npy"
    best_fit_lf_path = mcmc_root / f"{output_prefix}_best_fit_lf.csv"
    summary_path = mcmc_root / f"{output_prefix}_run_summary.json"

    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior samples file: {posterior_path}")
    if not chain_path.exists():
        raise FileNotFoundError(f"Missing chain file: {chain_path}")
    if not best_fit_lf_path.exists():
        raise FileNotFoundError(f"Missing best-fit LF file: {best_fit_lf_path}")

    posterior_df = pd.read_csv(posterior_path)
    chain = np.load(chain_path)
    best_fit_df = pd.read_csv(best_fit_lf_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    trace_path = figures_root / f"{output_prefix}_trace.png"
    _make_trace_plot(
        chain,
        INPUT_COLUMNS,
        trace_path,
        parameter_specs=INPUT_PARAMETER_SPECS,
        burn_in=int(summary.get("burn_in", 0)),
    )

    corner_path = figures_root / f"{output_prefix}_corner.png"
    if corner is not None:
        best_index = int(np.argmax(posterior_df["log_probability"].to_numpy(dtype=float)))
        best_theta = posterior_df.iloc[best_index][INPUT_COLUMNS].to_numpy(dtype=float)
        corner_fig = corner_with_log10_priors(
            corner,
            posterior_df,
            INPUT_COLUMNS,
            INPUT_PARAMETER_SPECS,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)

    sobral_labels = list(dict.fromkeys(best_fit_df["sobral_label"].tolist()))
    luminosity_by_label: dict[str, np.ndarray] = {}
    target_linear_by_label: dict[str, np.ndarray] = {}
    target_cov_linear_by_label: dict[str, np.ndarray] = {}
    target_by_label: dict[str, np.ndarray] = {}
    target_cov_by_label: dict[str, np.ndarray] = {}
    prediction_by_label: dict[str, np.ndarray] = {}
    prediction_std_by_label: dict[str, np.ndarray] = {}

    for sobral_label in sobral_labels:
        sub = best_fit_df.loc[best_fit_df["sobral_label"] == sobral_label].sort_values("bin_index")
        luminosity_by_label[sobral_label] = sub["luminosity_center"].to_numpy(dtype=float)
        target_linear_by_label[sobral_label] = sub["target_phi"].to_numpy(dtype=float)
        target_cov_linear_by_label[sobral_label] = np.diag(sub["target_phi_std"].to_numpy(dtype=float) ** 2)
        target_by_label[sobral_label] = sub["target_log10_lf"].to_numpy(dtype=float)
        target_cov_by_label[sobral_label] = np.diag(sub["target_log10_lf_std"].to_numpy(dtype=float) ** 2)
        prediction_by_label[sobral_label] = sub["prediction_log10_lf"].to_numpy(dtype=float)
        prediction_std_by_label[sobral_label] = sub["prediction_log10_lf_std"].to_numpy(dtype=float)

    best_fit_fig_path = figures_root / f"{output_prefix}_best_fit_lfs.png"
    _plot_best_fit_lfs(
        sobral_labels,
        luminosity_by_label,
        target_linear_by_label,
        target_cov_linear_by_label,
        target_by_label,
        target_cov_by_label,
        prediction_by_label,
        prediction_std_by_label,
        best_fit_fig_path,
    )

    print(trace_path)
    if corner is not None:
        print(corner_path)
    print(best_fit_fig_path)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    mcmc_root = campaign_root / args.mcmc_dir_name
    figures_root = campaign_root / args.figures_dir_name
    mcmc_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        _replot_from_saved_outputs(campaign_root, mcmc_root, figures_root, args.output_prefix)
        return

    if emcee is None:
        raise ModuleNotFoundError(
            "The 'emcee' package is required for run_halpha_pca_emulator_mcmc.py. "
            "Activate the galacticus-workspace environment or install emcee."
        )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    table = pd.read_csv(campaign_root / args.table_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    table = _limit_draws_per_eval(table, args.train_draws_per_eval)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, table[INPUT_COLUMNS].to_numpy(dtype=float))

    luminosity_by_label: dict[str, np.ndarray] = {}
    target_linear_by_label: dict[str, np.ndarray] = {}
    target_cov_linear_by_label: dict[str, np.ndarray] = {}
    target_by_label: dict[str, np.ndarray] = {}
    target_cov_by_label: dict[str, np.ndarray] = {}
    output_columns_by_label: dict[str, list[str]] = {}
    model_bundles: dict[str, dict[str, object]] = {}
    block_targets_linear: list[np.ndarray] = []
    block_covariances_linear: list[np.ndarray] = []
    output_slices: dict[str, slice] = {}

    start = 0
    for sobral_label in args.sobral_labels:
        output_columns = _output_columns(table, sobral_label)
        output_columns_by_label[sobral_label] = output_columns
        y_linear = table[output_columns].to_numpy(dtype=float)
        y_log10, floor, n_floored, n_clipped = _log10_with_floor(y_linear, args.min_log10_lf)
        print(
            f"{sobral_label}: training PCA emulator on {len(table)} rows x {len(output_columns)} bins "
            f"(floor={floor:.3e}, floored={n_floored}, clipped={n_clipped})",
            flush=True,
        )
        model_bundles[sobral_label] = _fit_pca_bundle(
            x,
            y_log10,
            sobral_label=sobral_label,
            pca_components=args.pca_components,
            pca_scaling=args.pca_scaling,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
        )

        luminosity, target_linear, target_cov_linear, target_log10, target_cov_log10 = _load_target_block(
            campaign_root, sobral_label
        )
        if len(output_columns) != len(target_log10):
            raise ValueError(
                f"{sobral_label}: emulator output has {len(output_columns)} bins but target has {len(target_log10)}."
            )
        luminosity_by_label[sobral_label] = luminosity
        target_linear_by_label[sobral_label] = target_linear
        target_cov_linear_by_label[sobral_label] = target_cov_linear
        target_by_label[sobral_label] = target_log10
        target_cov_by_label[sobral_label] = target_cov_log10
        block_targets_linear.append(target_linear)
        block_covariances_linear.append(target_cov_linear)
        output_slices[sobral_label] = slice(start, start + len(output_columns))
        start += len(output_columns)

    target_vector = np.concatenate(block_targets_linear)
    target_covariance = _block_diag(block_covariances_linear)

    posterior = HalphaPCAEmulatorPosterior(
        parameter_specs=INPUT_PARAMETER_SPECS,
        model_bundles=model_bundles,
        target_vector=target_vector,
        target_covariance=target_covariance,
        output_slices=output_slices,
    )

    if args.init_center == "prior_center":
        center_quantiles = np.full(len(INPUT_PARAMETER_SPECS), 0.5, dtype=float)
        center_theta = transform_from_prior_quantiles(INPUT_PARAMETER_SPECS, center_quantiles[None, :])[0]
    else:
        center_theta = _best_training_theta(table, target_by_label, output_columns_by_label, args.min_log10_lf)
        center_quantiles = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, center_theta[None, :])[0]

    rng = np.random.default_rng(args.seed)
    initial_quantiles = np.clip(
        center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(INPUT_COLUMNS))),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    initial_positions = transform_from_prior_quantiles(INPUT_PARAMETER_SPECS, initial_quantiles)

    sampler_kwargs = {"nwalkers": args.n_walkers, "ndim": len(INPUT_COLUMNS)}
    if args.n_processes > 1:
        from multiprocessing import Pool

        with Pool(processes=args.n_processes) as pool:
            sampler = emcee.EnsembleSampler(
                log_prob_fn=posterior.log_probability,
                pool=pool,
                vectorize=False,
                **sampler_kwargs,
            )
            sampler.run_mcmc(
                initial_positions,
                args.n_steps,
                progress=args.progress,
                skip_initial_state_check=True,
            )
            chain = sampler.get_chain()
            log_prob = sampler.get_log_prob()
            flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
            flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
            acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)
    else:
        sampler = emcee.EnsembleSampler(
            log_prob_fn=posterior.log_probability_batch,
            vectorize=True,
            **sampler_kwargs,
        )
        sampler.run_mcmc(
            initial_positions,
            args.n_steps,
            progress=args.progress,
            skip_initial_state_check=True,
        )
        chain = sampler.get_chain()
        log_prob = sampler.get_log_prob()
        flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
        acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)

    np.save(mcmc_root / f"{args.output_prefix}_chain.npy", chain)
    np.save(mcmc_root / f"{args.output_prefix}_log_prob.npy", log_prob)

    posterior_df = pd.DataFrame(flat_samples, columns=INPUT_COLUMNS)
    posterior_df["log_probability"] = flat_log_prob
    posterior_path = mcmc_root / f"{args.output_prefix}_posterior_samples.csv"
    posterior_df.to_csv(posterior_path, index=False)

    best_index = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_index]
    best_mean_linear, best_var_linear = posterior.predict(best_theta)
    best_std_linear = np.sqrt(best_var_linear)
    best_mean, best_var = posterior.predict_log10(best_theta)
    best_std = np.sqrt(best_var)

    prediction_rows: list[dict[str, object]] = []
    best_mean_by_label: dict[str, np.ndarray] = {}
    best_std_by_label: dict[str, np.ndarray] = {}
    best_mean_linear_by_label: dict[str, np.ndarray] = {}
    best_std_linear_by_label: dict[str, np.ndarray] = {}
    for sobral_label in args.sobral_labels:
        slc = output_slices[sobral_label]
        pred = best_mean[slc]
        pred_std = best_std[slc]
        pred_linear = best_mean_linear[slc]
        pred_linear_std = best_std_linear[slc]
        best_mean_by_label[sobral_label] = pred
        best_std_by_label[sobral_label] = pred_std
        best_mean_linear_by_label[sobral_label] = pred_linear
        best_std_linear_by_label[sobral_label] = pred_linear_std
        for bin_index, (luminosity, target, target_std, target_linear, target_linear_std, value, value_std, value_linear, value_linear_std) in enumerate(
            zip(
                luminosity_by_label[sobral_label],
                target_by_label[sobral_label],
                np.sqrt(np.diag(target_cov_by_label[sobral_label])),
                target_linear_by_label[sobral_label],
                np.sqrt(np.diag(target_cov_linear_by_label[sobral_label])),
                pred,
                pred_std,
                pred_linear,
                pred_linear_std,
                strict=True,
            )
        ):
            prediction_rows.append(
                {
                    "sobral_label": sobral_label,
                    "bin_index": bin_index,
                    "log10_luminosity_center": float(np.log10(luminosity)),
                    "luminosity_center": float(luminosity),
                    "target_phi": float(target_linear),
                    "target_phi_std": float(target_linear_std),
                    "target_log10_lf": float(target),
                    "target_log10_lf_std": float(target_std),
                    "prediction_phi": float(value_linear),
                    "prediction_phi_std": float(value_linear_std),
                    "prediction_log10_lf": float(value),
                    "prediction_log10_lf_std": float(value_std),
                }
            )
    prediction_df = pd.DataFrame(prediction_rows)
    prediction_path = mcmc_root / f"{args.output_prefix}_best_fit_lf.csv"
    prediction_df.to_csv(prediction_path, index=False)

    best_params_path = mcmc_root / f"{args.output_prefix}_best_fit_parameters.json"
    best_params_path.write_text(
        json.dumps(
            {name: float(value) for name, value in zip(INPUT_COLUMNS, best_theta, strict=True)},
            indent=2,
        )
        + "\n"
    )

    summary = {
        "campaign_root": str(campaign_root),
        "sobral_labels": args.sobral_labels,
        "train_draws_per_eval": args.train_draws_per_eval,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "pca_components": args.pca_components,
        "pca_scaling": args.pca_scaling,
        "min_log10_lf": args.min_log10_lf,
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "likelihood_space": "linear_phi",
        "n_walkers": args.n_walkers,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "seed": args.seed,
        "init_center": args.init_center,
        "init_quantile_sigma": args.init_quantile_sigma,
        "n_processes": args.n_processes,
        "progress": args.progress,
        "mean_acceptance_fraction": float(np.mean(acceptance_fraction)),
        "acceptance_fraction_by_walker": [float(value) for value in acceptance_fraction],
        "input_columns": INPUT_COLUMNS,
        "best_log_probability": float(flat_log_prob[best_index]),
        "best_theta": {name: float(value) for name, value in zip(INPUT_COLUMNS, best_theta, strict=True)},
        "best_prediction_mean_phi": {
            sobral_label: [float(value) for value in best_mean_linear_by_label[sobral_label]]
            for sobral_label in args.sobral_labels
        },
        "best_prediction_std_phi": {
            sobral_label: [float(value) for value in best_std_linear_by_label[sobral_label]]
            for sobral_label in args.sobral_labels
        },
        "best_prediction_mean_log10_phi": {
            sobral_label: [float(value) for value in best_mean_by_label[sobral_label]]
            for sobral_label in args.sobral_labels
        },
        "best_prediction_std_log10_phi": {
            sobral_label: [float(value) for value in best_std_by_label[sobral_label]]
            for sobral_label in args.sobral_labels
        },
        "pca_models": {
            sobral_label: {
                "n_components_actual": int(bundle["n_components_actual"]),
                "pca_scaling": bundle["pca_scaling"],
                "explained_variance_ratio": bundle["explained_variance_ratio"],
                "kernel_summaries": [component["kernel"] for component in bundle["component_models"]],
            }
            for sobral_label, bundle in model_bundles.items()
        },
        "target_dimension": int(len(target_vector)),
    }
    summary_path = mcmc_root / f"{args.output_prefix}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    trace_path = figures_root / f"{args.output_prefix}_trace.png"
    trace_tau = _make_trace_plot(
        chain,
        INPUT_COLUMNS,
        trace_path,
        parameter_specs=INPUT_PARAMETER_SPECS,
        burn_in=args.burn_in,
    )
    summary["trace_autocorrelation_time"] = trace_tau
    summary["trace_autocorrelation_time_burn_in"] = int(args.burn_in)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    corner_path = figures_root / f"{args.output_prefix}_corner.png"
    if corner is not None:
        corner_fig = corner_with_log10_priors(
            corner,
            posterior_df,
            INPUT_COLUMNS,
            INPUT_PARAMETER_SPECS,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)

    best_fit_fig_path = figures_root / f"{args.output_prefix}_best_fit_lfs.png"
    _plot_best_fit_lfs(
        args.sobral_labels,
        luminosity_by_label,
        target_linear_by_label,
        target_cov_linear_by_label,
        target_by_label,
        target_cov_by_label,
        best_mean_by_label,
        best_std_by_label,
        best_fit_fig_path,
    )

    print(posterior_path)
    print(prediction_path)
    print(best_params_path)
    print(summary_path)
    print(trace_path)
    if corner is not None:
        print(corner_path)
    print(best_fit_fig_path)


if __name__ == "__main__":
    main()
