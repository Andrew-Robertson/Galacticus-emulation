from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_checked_in_figure_data_rebuilds_supported_paper_figures(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repository_root / "paper/figure_scripts/rebuild_paper_figures.py"),
            "--from-figure-data",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repository_root,
        check=True,
        env={**os.environ, "PANDAS_COPY_ON_WRITE": "1"},
    )

    expected_pdfs = [
        "smf_z0_pca99_holdout_curves_highlight_bins.pdf",
        "smf_z0_pca99_parity_highlight_bins_random_quarter.pdf",
        "smf_z0_pca_threshold_rmse_r2_vs_training_size.pdf",
        "final_map_validation_all_observables_4x3.pdf",
    ]
    for filename in expected_pdfs:
        output = tmp_path / filename
        assert output.is_file()
        assert output.stat().st_size > 0
