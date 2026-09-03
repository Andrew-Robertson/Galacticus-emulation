from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shlex
import subprocess
import sys

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS = (
    REPO_ROOT
    / "runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced"
    / "automatedPipeline_transformedParams_finalPaper_definitive"
    / "MCMCs_production_from_exploratory_MAP"
    / "standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
    / "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr_mcmc_results.hdf5"
)


PARAMETER_GROUPS = [
    (
        "Star formation and stellar populations",
        [
            (
                "diskSFRFrequencyNorm",
                r"\(\nu_{\rm SF,d}/\mathrm{Gyr}^{-1}\)",
                1.0e9,
            ),
            ("spheroidSFREfficiency", r"\(\epsilon_{\star,{\rm sph}}\)", 1.0),
            ("stellarPopulationMetalYield", r"\(y_Z\)", 1.0),
        ],
    ),
    (
        "Stellar feedback",
        [
            (
                "diskVelocityCharacteristic",
                r"\(V_{\rm out,d}/\mathrm{km\,s^{-1}}\)",
                1.0,
            ),
            ("diskExponent", r"\(\alpha_{\rm out,d}\)", 1.0),
            (
                "spheroidVelocityCharacteristic",
                r"\(V_{\rm out,sph}/\mathrm{km\,s^{-1}}\)",
                1.0,
            ),
        ],
    ),
    (
        "Gas cooling and recycling",
        [
            ("coolingMultiplier", r"\(f_{\rm cool}\)", 1.0),
            ("coreRadiusOverVirialRadius", r"\(r_{\rm core}/R_{\rm vir}\)", 1.0),
            ("henriquesGamma", r"\(\gamma_{\rm reinc}\)", 1.0),
            ("henriquesDelta1", r"\(\delta_{\rm reinc,1}\)", 1.0),
            ("henriquesDelta2", r"\(\delta_{\rm reinc,2}\)", 1.0),
        ],
    ),
    (
        "Mergers and disk instabilities",
        [
            ("massRatioMajorMerger", r"\(f_{\rm major}\)", 1.0),
            ("energyOrbital", r"\(f_{\rm orbit}\)", 1.0),
            (
                "barFractionAngularMomentumRetainedSpheroid",
                r"\(f_{j,{\rm sph}}^{\rm bar}\)",
                1.0,
            ),
        ],
    ),
    (
        "Supermassive black holes",
        [
            (
                "blackHoleSeedMass",
                r"\(M_{\rm BH,seed}/(10^4\mathrm{M_\odot})\)",
                1.0e-4,
            ),
            ("BHefficiencyRadioMode", r"\(\epsilon_{\rm BH,radio}\)", 1.0),
            ("BHefficiencyWind", r"\(\epsilon_{\rm BH,wind}\)", 1.0),
            ("bondiEnhancementSpheroid", r"\(\alpha_{\rm Bondi,sph}\)", 1.0),
            ("bondiEnhancementHotHalo", r"\(\alpha_{\rm Bondi,hot}\)", 1.0),
            ("thinDiskMaximum", r"\(x_{\rm thin,max}\)", 1.0),
        ],
    ),
    (
        "Dust attenuation",
        [
            ("delta_0", r"\(\delta_0\)", 1.0),
            ("delta_M", r"\(\delta_M\)", 1.0),
            ("delta_z", r"\(\delta_z\)", 1.0),
            ("delta_Mz", r"\(\delta_{Mz}\)", 1.0),
            ("attenuation_scatter", r"\(\sigma_A\)", 1.0),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final calibrated-parameter posterior summary table."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to the final *_mcmc_results.hdf5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex path. If omitted, write the table to stdout.",
    )
    parser.add_argument("--value-sigfigs", type=int, default=3)
    parser.add_argument("--prior-percent-decimals", type=int, default=1)
    parser.add_argument("--tabcolsep", default="4.2pt")
    parser.add_argument("--arraystretch", default="1.10")
    parser.add_argument("--block-space", default="1.4ex")
    parser.add_argument(
        "--include-provenance-comments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include LaTeX comments recording the command, source file, and git state.",
    )
    parser.add_argument(
        "--include-git-status-comments",
        action="store_true",
        help="Include the full short git status in the generated LaTeX comments.",
    )
    return parser.parse_args()


def normal_cdf(value: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(value, dtype=float)
    result = 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))
    if np.isscalar(value):
        return float(result)
    return result


def prior_cdf(
    values: np.ndarray,
    *,
    prior_type: str,
    lower: float,
    upper: float,
    x0: float,
    mean: float,
    sigma: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if prior_type == "TruncatedLogNormalPrior":
        lower_z = (np.log(lower) - np.log(x0)) / sigma
        upper_z = (np.log(upper) - np.log(x0)) / sigma
        z = (np.log(values) - np.log(x0)) / sigma
        return (normal_cdf(z) - normal_cdf(lower_z)) / (normal_cdf(upper_z) - normal_cdf(lower_z))
    if prior_type == "UniformPrior":
        return (values - lower) / (upper - lower)
    if prior_type == "NormalPrior":
        return normal_cdf((values - mean) / sigma)
    if prior_type == "TruncatedNormalPrior":
        lower_z = (lower - mean) / sigma
        upper_cdf = 1.0 if not np.isfinite(upper) else normal_cdf((upper - mean) / sigma)
        z = (values - mean) / sigma
        return (normal_cdf(z) - normal_cdf(lower_z)) / (upper_cdf - normal_cdf(lower_z))
    raise ValueError(f"Unsupported prior type: {prior_type}")


def format_sigfig(value: float, sigfigs: int) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if exponent >= sigfigs or exponent < -4:
        mantissa = value / (10.0**exponent)
        decimals = max(sigfigs - 1, 0)
        return rf"{mantissa:.{decimals}f}\times10^{{{exponent}}}"
    decimals = max(sigfigs - exponent - 1, 0)
    return f"{value:.{decimals}f}"


def format_prior_percentile(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def load_summary(results: Path) -> dict[str, dict[str, float | np.ndarray]]:
    with h5py.File(results, "r") as handle:
        short_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["parameters/short_names"][()]
        ]
        prior_types = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["parameters/prior_type"][()]
        ]
        lower = handle["parameters/prior_lower"][()]
        upper = handle["parameters/prior_upper"][()]
        x0 = handle["parameters/prior_x0"][()]
        mean = handle["parameters/prior_mean"][()]
        sigma = handle["parameters/prior_sigma"][()]
        theta_map = handle["map/theta"][()]
        q_map = handle["map/quantile"][()] * 100.0
        samples = handle["posterior/samples_flat"][()]

    summary: dict[str, dict[str, float | np.ndarray]] = {}
    for index, short_name in enumerate(short_names):
        q_posterior = (
            np.percentile(
                prior_cdf(
                    samples[:, index],
                    prior_type=prior_types[index],
                    lower=float(lower[index]),
                    upper=float(upper[index]),
                    x0=float(x0[index]),
                    mean=float(mean[index]),
                    sigma=float(sigma[index]),
                ),
                [16.0, 50.0, 84.0],
            )
            * 100.0
        )
        summary[short_name] = {
            "map": float(theta_map[index]),
            "posterior": np.percentile(samples[:, index], [16.0, 50.0, 84.0]),
            "q_map": float(q_map[index]),
            "q_posterior": q_posterior,
        }
    return summary


def git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as error:
        return f"<git unavailable: {error}>"
    output = completed.stdout.strip()
    error_output = completed.stderr.strip()
    if completed.returncode != 0:
        return error_output or f"<git {' '.join(args)} failed with code {completed.returncode}>"
    return output


def provenance_comments(results: Path, *, include_git_status: bool) -> list[str]:
    lines = [
        "% Generated by paper/figure_scripts/generate_final_calibrated_parameters_table.py",
        f"% created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"% command: {shlex.join([sys.executable, *sys.argv])}",
        f"% cwd: {Path.cwd()}",
        f"% source_hdf5: {results.expanduser().resolve()}",
        f"% git_head: {git_output(['rev-parse', 'HEAD'])}",
    ]
    if include_git_status:
        git_status = git_output(["status", "--short"])
        lines.append("% git_status_short:")
        lines.extend(f"%   {line}" for line in (git_status.splitlines() or ["<clean>"]))
    lines.append("")
    return lines


def render_table(args: argparse.Namespace) -> str:
    results = args.results.expanduser().resolve()
    summary = load_summary(results)
    lines: list[str] = []
    if args.include_provenance_comments:
        lines.extend(provenance_comments(results, include_git_status=args.include_git_status_comments))
    lines.extend(
        [
            rf"\providecommand{{\paramblockspace}}{{\addlinespace[{args.block_space}]}}",
            "",
            r"\begin{table*}",
            r"\centering",
            r"\scriptsize",
            rf"\setlength{{\tabcolsep}}{{{args.tabcolsep}}}",
            rf"\renewcommand{{\arraystretch}}{{{args.arraystretch}}}",
            r"\caption{",
            r"Posterior summary for the final calibrated model.",
            r"The columns \(p_{16}\), \(p_{50}\), and \(p_{84}\) give the 16th, 50th, and",
            r"84th percentiles of the marginalized posterior.",
            r"The \(q\) columns give the corresponding locations in the adopted prior,",
            r"expressed as percentiles on a 0--100 scale.",
            r"Units for dimensional parameters are shown in the parameter column.",
            r"}",
            r"\label{tab:final-calibrated-parameters}",
            r"\begin{tabular}{@{}l @{\hspace{2.1em}} r @{\hspace{2.5em}} rrr @{\hspace{3.1em}} rrrr@{}}",
            r"\toprule",
            r"Parameter & MAP & \multicolumn{3}{c}{Posterior} & \multicolumn{4}{c}{Prior percentile} \\",
            r"\cmidrule(l{1.0em}r{2.0em}){3-5}",
            r"\cmidrule(l{0.0em}r{0.0em}){6-9}",
            r"& & \(p_{16}\) & \(p_{50}\) & \(p_{84}\) & \(q_{\rm MAP}\) & \(q_{16}\) & \(q_{50}\) & \(q_{84}\) \\",
            r"\midrule",
        ]
    )
    missing = [
        name
        for _, parameters in PARAMETER_GROUPS
        for name, _, _ in parameters
        if name not in summary
    ]
    if missing:
        raise KeyError(f"Parameter(s) missing from {results}: {', '.join(missing)}")

    for group_index, (group_name, parameters) in enumerate(PARAMETER_GROUPS):
        if group_index > 0:
            lines.append("")
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\normalfont {group_name}}} \\")
        for name, label, scale in parameters:
            row = summary[name]
            posterior = np.asarray(row["posterior"], dtype=float) * scale
            q_posterior = np.asarray(row["q_posterior"], dtype=float)
            value_cells = [
                format_sigfig(float(row["map"]) * scale, args.value_sigfigs),
                *(format_sigfig(value, args.value_sigfigs) for value in posterior),
            ]
            q_cells = [
                format_prior_percentile(float(row["q_map"]), args.prior_percent_decimals),
                *(format_prior_percentile(value, args.prior_percent_decimals) for value in q_posterior),
            ]
            line_cells = [label, *(rf"\({cell}\)" for cell in value_cells), *q_cells]
            lines.append(" & ".join(line_cells) + r" \\")
        if group_index < len(PARAMETER_GROUPS) - 1:
            lines.append(r"\paramblockspace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    table = render_table(args)
    if args.output is None:
        sys.stdout.write(table)
        return
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(output)


if __name__ == "__main__":
    main()
