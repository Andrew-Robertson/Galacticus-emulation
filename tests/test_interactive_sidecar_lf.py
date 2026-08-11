from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from galacticus_emu.interactive_sidecar_lf import (
    bundle_meta,
    load_sidecar_lf_bundle,
    predict_sidecar_lf_bundle,
)
from galacticus_emu.lhs import NormalPrior, ParameterSpec, UniformPrior


class ConstantModel:
    def __init__(self, value: float, std: float):
        self.value = float(value)
        self.std = float(std)

    def predict(self, x, return_std=False):
        values = np.full(x.shape[0], self.value, dtype=float)
        if return_std:
            return values, np.full(x.shape[0], self.std, dtype=float)
        return values


def _write_preview_table(root: Path) -> None:
    sidecar_dir = root / "evaluations" / "eval0000" / "emission_line_dust"
    sidecar_dir.mkdir(parents=True)
    (root / "samples.csv").write_text(
        "diskVelocityCharacteristic,delta_0\n"
        "90.0,-0.2\n"
        "130.0,0.4\n"
    )
    (sidecar_dir / "emission_line_dust_lf_emulator_table.csv").write_text(
        "evaluation_id,dust_draw_index,phi_bin0,phi_bin1\n"
        "0,0,1e-4,1e-5\n"
        "0,1,1e-6,0.0\n"
    )


def _bundle(root: Path) -> dict:
    return {
        "bundle_type": "sidecar_lf_pca_gp",
        "campaign_root": str(root),
        "sidecar_dir": "emission_line_dust",
        "table_filename": "emission_line_dust_lf_emulator_table.csv",
        "long_filename": "emission_line_dust_lf_long.csv",
        "input_columns": ["diskVelocityCharacteristic_quantile", "delta_0"],
        "parameter_names": ["diskVelocityCharacteristic", "delta_0"],
        "parameter_specs": [
            ParameterSpec(
                path="diskVelocityCharacteristic",
                short_name="diskVelocityCharacteristic",
                prior=UniformPrior(lower=80.0, upper=140.0),
            ),
            ParameterSpec(
                path="delta_0",
                short_name="delta_0",
                prior=NormalPrior(mean=0.0, sigma=0.5),
            ),
        ],
        "slow_count": 1,
        "observables": {
            "halpha_sobral_z1": {
                "observable_key": "halpha_sobral",
                "observable": "halpha_sobral",
                "sample_label": "Z1",
                "output_columns": ["phi_bin0", "phi_bin1"],
                "x_plot": np.asarray([41.0, 42.0]),
                "target_plot": np.asarray([-4.0, -5.0]),
                "target_sigma_plot": None,
                "models": [ConstantModel(value=0.5, std=0.1)],
                "y_means": np.asarray([0.0]),
                "y_stds": np.asarray([1.0]),
                "pca_components_actual": 1,
                "pca_variance_threshold": 0.99,
                "pca_explained_variance_ratio": np.asarray([1.0]),
                "pca_mean": np.asarray([0.0, 0.0]),
                "pca_components_matrix": np.asarray([[1.0, -1.0]]),
                "preprocessor_mean": np.asarray([1.0, 2.0]),
                "preprocessor_scale": np.asarray([2.0, 3.0]),
                "kernels": ["constant-test-kernel"],
            }
        },
        "observable_groups": {"halpha_sobral": ["halpha_sobral_z1"]},
        "n_training_rows": 2,
        "training_options": {
            "dust_draw_index": [0],
            "max_dust_draws_per_eval": None,
            "min_log10_lf": -8.0,
        },
    }


class InteractiveSidecarLFTest(unittest.TestCase):
    def test_load_predict_and_metadata_for_sidecar_lf_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_preview_table(root)
            bundle_path = root / "sidecar_lf_demo_bundle.joblib"
            joblib.dump(_bundle(root), bundle_path)

            bundle = load_sidecar_lf_bundle(bundle_path)
            prediction = predict_sidecar_lf_bundle(
                bundle,
                {
                    "diskVelocityCharacteristic": 110.0,
                    "delta_0": 0.1,
                },
            )

            payload = prediction["halpha_sobral_z1"]
            np.testing.assert_allclose(payload["x_plot"], [41.0, 42.0])
            np.testing.assert_allclose(payload["y_pred_plot"], [2.0, 0.5])
            np.testing.assert_allclose(payload["y_std_plot"], [0.2, 0.3])

            meta = bundle_meta(bundle, training_preview_rows=1)
            observable = meta["observables"]["halpha_sobral_z1"]
            self.assertEqual(meta["bundle_type"], "sidecar_lf_pca_gp")
            self.assertEqual(meta["observable_groups"], {"halpha_sobral": ["halpha_sobral_z1"]})
            self.assertEqual(observable["target_label"], "Sobral et al. (2013)")
            self.assertEqual(observable["x_axis_label"], r"$\log_{10}(L_{\mathrm{H}\alpha}/\mathrm{erg\,s}^{-1})$")
            self.assertEqual(len(observable["y_train_preview"]), 1)
            self.assertEqual(meta["input_ranges"]["diskVelocityCharacteristic"]["min"], 90.0)
            self.assertEqual(meta["input_ranges"]["diskVelocityCharacteristic"]["max"], 130.0)


if __name__ == "__main__":
    unittest.main()
