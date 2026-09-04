from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp


def test_mean_only_export_preserves_gp_mean() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "scripts" / "export_mean_only_emulator_bundle.py"
    spec = importlib.util.spec_from_file_location("mean_only_export", script_path)
    assert spec is not None and spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    x_train = np.linspace(0.0, 1.0, 8)[:, None]
    y_train = np.sin(2.0 * np.pi * x_train[:, 0])
    model, y_mean, y_std = fit_scaled_gp(
        x_train,
        y_train,
        n_restarts_optimizer=0,
        optimize_hyperparameters=False,
    )
    x_test = np.asarray([[0.15], [0.55], [0.9]])
    expected_mean, _expected_std = predict_scaled_gp(model, y_mean, y_std, x_test)

    bundle = {"observables": {"example": {"models": [model]}}}
    exporter.strip_predictive_uncertainty(bundle)
    actual_mean, actual_std = predict_scaled_gp(
        model,
        y_mean,
        y_std,
        x_test,
        return_std=False,
    )

    np.testing.assert_array_equal(actual_mean, expected_mean)
    np.testing.assert_array_equal(actual_std, np.zeros_like(actual_mean))
    assert bundle["supports_predictive_uncertainty"] is False
    assert not hasattr(model, "L_")


def test_export_embeds_best_fit_values(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "scripts" / "export_mean_only_emulator_bundle.py"
    spec = importlib.util.spec_from_file_location("mean_only_export", script_path)
    assert spec is not None and spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"best_theta": {"alpha": 1.25, "beta": -0.5}}))
    bundle = {"input_columns": ["alpha", "beta"]}

    exporter.embed_best_fit(bundle, summary_path)

    assert bundle["best_fit_params"] == {"alpha": 1.25, "beta": -0.5}
