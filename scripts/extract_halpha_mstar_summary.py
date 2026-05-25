from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import h5py
import numpy as np


SOBRAL_OUTPUTS = [
    ("Output16", 0.40, "Z1", "luminosityFunctionHalphaSobral2013HiZELSZ1"),
    ("Output13", 0.84, "Z2", "luminosityFunctionHalphaSobral2013HiZELSZ2"),
    ("Output10", 1.47, "Z3", "luminosityFunctionHalphaSobral2013HiZELSZ3"),
    ("Output6", 2.23, "Z4", "luminosityFunctionHalphaSobral2013HiZELSZ4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a reduced Halpha summary from a Galacticus output, storing "
            "2D counts in log10(L_Halpha)-log10(Mstar) and Sobral-bin marginals."
        )
    )
    parser.add_argument("input_hdf5", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--group-name", default="emissionLineSummaries")
    parser.add_argument("--line-name", default="halphaSobral")
    parser.add_argument("--luminosity-min", type=float, default=39.5)
    parser.add_argument("--luminosity-max", type=float, default=43.5)
    parser.add_argument("--luminosity-bin-width", type=float, default=0.1)
    parser.add_argument("--stellar-mass-min", type=float, default=7.5)
    parser.add_argument("--stellar-mass-max", type=float, default=12.5)
    parser.add_argument("--stellar-mass-bin-width", type=float, default=0.25)
    parser.add_argument("--use-centrals-only", action="store_true")
    parser.add_argument(
        "--fill-missing-node-weights-with-median",
        action="store_true",
        help=(
            "Salvage Galacticus files affected by the merger-tree weight race condition by filling "
            "unassigned node weights with the median finite mergerTreeWeight. Default is strict/error."
        ),
    )
    return parser.parse_args()


def _edges(lower: float, upper: float, width: float) -> np.ndarray:
    n_steps = int(round((upper - lower) / width))
    if not np.isclose(lower + n_steps * width, upper):
        raise ValueError("Grid edges must be evenly spaced by the requested width")
    return lower + width * np.arange(n_steps + 1, dtype=float)


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Need at least two centers to infer bin edges")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def _read_total_halpha_and_mstar(node_data: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    luminosity = (
        np.asarray(node_data["luminosityEmissionLineAGN:balmerAlpha6565"][...], dtype=float)
        + np.asarray(node_data["luminosityEmissionLineDisk:balmerAlpha6565"][...], dtype=float)
        + np.asarray(node_data["luminosityEmissionLineSpheroid:balmerAlpha6565"][...], dtype=float)
    )
    stellar_mass = (
        np.asarray(node_data["diskMassStellar"][...], dtype=float)
        + np.asarray(node_data["spheroidMassStellar"][...], dtype=float)
    )
    return luminosity, stellar_mass


def _read_weights(node_data: h5py.Group) -> np.ndarray:
    if "nodeSubsamplingWeight" in node_data:
        return np.asarray(node_data["nodeSubsamplingWeight"][...], dtype=float)
    first_dataset = next(iter(node_data.values()))
    return np.ones(first_dataset.shape[0], dtype=float)


def _read_tree_weights(
    output_group: h5py.Group,
    n_nodes: int,
    *,
    fill_missing_with_median: bool = False,
) -> np.ndarray:
    if not all(name in output_group for name in ["mergerTreeStartIndex", "mergerTreeCount", "mergerTreeWeight"]):
        return np.ones(n_nodes, dtype=float)
    tree_weights = np.asarray(output_group["mergerTreeWeight"][...], dtype=float)
    tree_counts = np.asarray(output_group["mergerTreeCount"][...], dtype=int)
    tree_starts = np.asarray(output_group["mergerTreeStartIndex"][...], dtype=int)
    node_weights = np.zeros(n_nodes, dtype=float)
    assigned = np.zeros(n_nodes, dtype=bool)
    for start, count, weight in zip(tree_starts, tree_counts, tree_weights, strict=True):
        stop = start + count
        if start < 0 or stop > n_nodes:
            raise ValueError(f"Merger-tree node range [{start}:{stop}] is outside nodeData length {n_nodes}")
        node_weights[start:stop] = weight
        assigned[start:stop] = True
    if np.any(~assigned):
        missing_count = int(np.count_nonzero(~assigned))
        if not fill_missing_with_median:
            raise ValueError("Some nodes were not assigned a merger-tree weight")
        finite_weights = tree_weights[np.isfinite(tree_weights)]
        if finite_weights.size == 0:
            raise ValueError("Some nodes were not assigned a merger-tree weight and no finite median weight is available")
        fill_value = float(np.median(finite_weights))
        node_weights[~assigned] = fill_value
        warnings.warn(
            f"Filled {missing_count} unassigned node weight(s) in {output_group.name} "
            f"with median merger-tree weight {fill_value:.6g}",
            RuntimeWarning,
            stacklevel=2,
        )
    return node_weights


def _summary_for_output(
    handle: h5py.File,
    output_name: str,
    analysis_name: str,
    luminosity_edges: np.ndarray,
    stellar_mass_edges: np.ndarray,
    *,
    use_centrals_only: bool,
    fill_missing_node_weights_with_median: bool,
) -> dict[str, np.ndarray | float | int | str]:
    output_group = handle[f"/Outputs/{output_name}"]
    node_data = output_group["nodeData"]
    luminosity, stellar_mass = _read_total_halpha_and_mstar(node_data)
    subsampling_weights = _read_weights(node_data)
    tree_weights = _read_tree_weights(
        output_group,
        luminosity.size,
        fill_missing_with_median=fill_missing_node_weights_with_median,
    )
    weights = subsampling_weights * tree_weights
    is_central = np.asarray(node_data["nodeIsIsolated"][...], dtype=int) == 1 if "nodeIsIsolated" in node_data else np.ones_like(weights, dtype=bool)

    valid = np.isfinite(luminosity) & np.isfinite(stellar_mass) & np.isfinite(weights) & (luminosity > 0.0) & (stellar_mass > 0.0)
    if use_centrals_only:
        valid &= is_central

    log_luminosity = np.log10(luminosity[valid])
    log_stellar_mass = np.log10(stellar_mass[valid])
    used_weights = weights[valid]
    in_fine_grid = (
        (log_luminosity >= luminosity_edges[0])
        & (log_luminosity < luminosity_edges[-1])
        & (log_stellar_mass >= stellar_mass_edges[0])
        & (log_stellar_mass < stellar_mass_edges[-1])
    )

    counts_2d, _, _ = np.histogram2d(
        log_luminosity,
        log_stellar_mass,
        bins=[luminosity_edges, stellar_mass_edges],
        weights=used_weights,
    )
    counts_2d_unweighted, _, _ = np.histogram2d(
        log_luminosity,
        log_stellar_mass,
        bins=[luminosity_edges, stellar_mass_edges],
    )
    intrinsic_lf = counts_2d.sum(axis=1)

    sobral = handle[f"/analyses/{analysis_name}"]
    sobral_centers_linear = np.asarray(sobral["luminosity"][...], dtype=float)
    sobral_centers = np.log10(sobral_centers_linear)
    sobral_edges = _bin_edges_from_centers(sobral_centers)
    sobral_counts, _ = np.histogram(log_luminosity, bins=sobral_edges, weights=used_weights)
    sobral_counts_unweighted, _ = np.histogram(log_luminosity, bins=sobral_edges)
    in_sobral_l_grid = (log_luminosity >= sobral_edges[0]) & (log_luminosity < sobral_edges[-1])

    return {
        "counts_2d": counts_2d,
        "counts_2d_unweighted": counts_2d_unweighted.astype(np.int64),
        "intrinsic_lf_fine": intrinsic_lf,
        "sobral_bin_centers_log10_luminosity": sobral_centers,
        "sobral_bin_edges_log10_luminosity": sobral_edges,
        "intrinsic_lf_sobral_bins": sobral_counts,
        "intrinsic_lf_sobral_bins_unweighted": sobral_counts_unweighted.astype(np.int64),
        "sobral_analysis_lf": np.asarray(sobral["luminosityFunction"][...], dtype=float),
        "sobral_target_lf": np.asarray(sobral["luminosityFunctionTarget"][...], dtype=float),
        "sobral_analysis_lf_covariance": np.asarray(sobral["luminosityFunctionCovariance"][...], dtype=float),
        "sobral_target_lf_covariance": np.asarray(sobral["luminosityFunctionCovarianceTarget"][...], dtype=float),
        "n_objects_total": np.int64(luminosity.size),
        "n_objects_used": np.int64(log_luminosity.size),
        "n_objects_in_fine_grid": np.int64(np.count_nonzero(in_fine_grid)),
        "n_objects_in_sobral_l_grid": np.int64(np.count_nonzero(in_sobral_l_grid)),
        "sum_weights_used": float(np.sum(used_weights)),
        "sum_weights_in_fine_grid": float(np.sum(used_weights[in_fine_grid])),
        "sum_weights_in_sobral_l_grid": float(np.sum(used_weights[in_sobral_l_grid])),
        "mean_tree_weight": float(np.mean(tree_weights)),
        "median_tree_weight": float(np.median(tree_weights)),
        "output_time": float(output_group.attrs["outputTime"]),
    }


def main() -> None:
    args = parse_args()
    input_hdf5 = args.input_hdf5.resolve()
    output_hdf5 = args.output.resolve() if args.output is not None else input_hdf5.with_name(input_hdf5.stem + "_halpha_summary.hdf5")

    luminosity_edges = _edges(args.luminosity_min, args.luminosity_max, args.luminosity_bin_width)
    stellar_mass_edges = _edges(args.stellar_mass_min, args.stellar_mass_max, args.stellar_mass_bin_width)
    luminosity_centers = _centers(luminosity_edges)
    stellar_mass_centers = _centers(stellar_mass_edges)

    with h5py.File(input_hdf5, "r") as source, h5py.File(output_hdf5, "w") as destination:
        root = destination.create_group(args.group_name)
        root.attrs["sourceFile"] = str(input_hdf5)
        root.attrs["lineName"] = args.line_name
        root.attrs["selection"] = "centrals" if args.use_centrals_only else "all"
        root.attrs["fillMissingNodeWeightsWithMedian"] = bool(args.fill_missing_node_weights_with_median)
        root.attrs["luminosityDatasetDefinition"] = (
            "luminosityEmissionLineAGN:balmerAlpha6565 + "
            "luminosityEmissionLineDisk:balmerAlpha6565 + "
            "luminosityEmissionLineSpheroid:balmerAlpha6565"
        )
        root.attrs["stellarMassDatasetDefinition"] = "diskMassStellar + spheroidMassStellar"
        root.create_dataset("log10_luminosity_edges", data=luminosity_edges)
        root.create_dataset("log10_luminosity_centers", data=luminosity_centers)
        root.create_dataset("log10_stellar_mass_edges", data=stellar_mass_edges)
        root.create_dataset("log10_stellar_mass_centers", data=stellar_mass_centers)

        for output_name, redshift, label, analysis_name in SOBRAL_OUTPUTS:
            group = root.create_group(f"{args.line_name}{label}")
            group.attrs["outputName"] = output_name
            group.attrs["redshift"] = redshift
            group.attrs["sobralAnalysisName"] = analysis_name
            summary = _summary_for_output(
                source,
                output_name,
                analysis_name,
                luminosity_edges,
                stellar_mass_edges,
                use_centrals_only=args.use_centrals_only,
                fill_missing_node_weights_with_median=args.fill_missing_node_weights_with_median,
            )
            for key, value in summary.items():
                group.create_dataset(key, data=value)

    print(output_hdf5)


if __name__ == "__main__":
    main()
