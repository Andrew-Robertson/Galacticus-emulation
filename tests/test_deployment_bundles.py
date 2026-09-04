from __future__ import annotations

from pathlib import Path

import numpy as np

from galacticus_emu.interactive_observables import (
    bundle_meta as observables_bundle_meta,
    load_observables_bundle,
    predict_observables_bundle,
)
from galacticus_emu.interactive_sidecar_lf import (
    bundle_meta as sidecar_bundle_meta,
    load_sidecar_lf_bundle,
    predict_sidecar_lf_bundle,
)


def test_checked_in_paper_bundles_are_self_contained() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    bundle_root = repository_root / "demo_emulators" / "paper"
    standard = load_observables_bundle(
        bundle_root / "standard_observables_mean_only.joblib"
    )
    sidecar = load_sidecar_lf_bundle(
        bundle_root / "halpha_sobral_log_error_mean_only.joblib"
    )

    for bundle, metadata, predict in (
        (standard, observables_bundle_meta(standard), predict_observables_bundle),
        (sidecar, sidecar_bundle_meta(sidecar), predict_sidecar_lf_bundle),
    ):
        assert metadata["supports_predictive_uncertainty"] is False
        assert metadata["best_fit_params"]
        assert metadata["galacticus_default_params"]
        assert metadata["n_training_rows"] == 1024
        assert all(
            observable["y_train_preview"]
            for observable in metadata["observables"].values()
        )

        predictions = predict(bundle, metadata["best_fit_params"])
        for prediction in predictions.values():
            assert np.isfinite(prediction["y_pred_plot"]).all()
            np.testing.assert_array_equal(
                prediction["y_std_plot"],
                np.zeros_like(prediction["y_std_plot"]),
            )
