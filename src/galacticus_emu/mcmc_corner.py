from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .lhs import ParameterSpec, TruncatedLogNormalPrior


def transform_corner_samples(
    samples: pd.DataFrame,
    labels: Sequence[str],
    parameter_specs: Sequence[ParameterSpec],
    truths: np.ndarray | Sequence[float] | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray | None]:
    transformed = samples.loc[:, list(labels)].copy()
    plot_labels = [str(label) for label in labels]
    transformed_truths = None if truths is None else np.asarray(truths, dtype=float).copy()

    for index, spec in enumerate(parameter_specs):
        if not isinstance(spec.prior, TruncatedLogNormalPrior):
            continue
        name = plot_labels[index]
        if np.any(transformed[name].to_numpy(dtype=float) <= 0.0):
            raise ValueError(f"Cannot log10-transform non-positive samples for parameter '{name}'")
        transformed[name] = np.log10(transformed[name].to_numpy(dtype=float))
        if transformed_truths is not None:
            if transformed_truths[index] <= 0.0:
                raise ValueError(f"Cannot log10-transform non-positive truth for parameter '{name}'")
            transformed_truths[index] = np.log10(transformed_truths[index])
        plot_labels[index] = f"log10({name})"
        transformed = transformed.rename(columns={name: plot_labels[index]})

    return transformed, plot_labels, transformed_truths


def corner_with_log10_priors(
    corner_module,
    samples: pd.DataFrame,
    labels: Sequence[str],
    parameter_specs: Sequence[ParameterSpec],
    *,
    truths: np.ndarray | Sequence[float] | None = None,
    **kwargs,
):
    plot_samples, plot_labels, plot_truths = transform_corner_samples(
        samples,
        labels,
        parameter_specs,
        truths=truths,
    )
    return corner_module.corner(plot_samples[plot_labels], labels=plot_labels, truths=plot_truths, **kwargs)
