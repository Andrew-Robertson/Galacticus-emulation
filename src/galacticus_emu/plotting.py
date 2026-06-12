from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def finite_limits_from_values(
    *values: np.ndarray | Iterable[float] | None,
    margin_fraction: float = 0.08,
    min_margin: float = 0.15,
) -> tuple[float, float] | None:
    finite_arrays = []
    for value in values:
        if value is None:
            continue
        array = np.asarray(value, dtype=float).ravel()
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_arrays.append(finite)
    if not finite_arrays:
        return None
    finite = np.concatenate(finite_arrays)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        lower -= min_margin
        upper += min_margin
    else:
        margin = max(margin_fraction * (upper - lower), min_margin)
        lower -= margin
        upper += margin
    return lower, upper


def set_ylim_from_values(
    axis,
    *values: np.ndarray | Iterable[float] | None,
    margin_fraction: float = 0.08,
    min_margin: float = 0.15,
) -> tuple[float, float] | None:
    limits = finite_limits_from_values(
        *values,
        margin_fraction=margin_fraction,
        min_margin=min_margin,
    )
    if limits is not None:
        axis.set_ylim(*limits)
    return limits
