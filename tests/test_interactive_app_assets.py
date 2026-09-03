from __future__ import annotations

import importlib.util
import os
from pathlib import Path

APP_ENVIRONMENT_VARIABLES = [
    "GALACTICUS_EMU_PROJECT_ROOT",
    "INTERACTIVE_SMF_ENABLED",
    "INTERACTIVE_HALPHA_ENABLED",
    "INTERACTIVE_OBSERVABLES_ENABLED",
    "INTERACTIVE_OBSERVABLES_BUNDLE_PATH",
    "INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH",
    "INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS",
    "INTERACTIVE_SIDECAR_LF_ENABLED",
    "INTERACTIVE_SIDECAR_LF_BUNDLE_PATH",
    "INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH",
    "INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS",
]


def _track_app_environment(monkeypatch) -> None:
    for variable in APP_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_hosted_demo_uses_checked_in_observable_bundle(monkeypatch, tmp_path) -> None:
    _track_app_environment(monkeypatch)
    repository_root = Path(__file__).resolve().parents[1]
    launcher_path = repository_root / "scripts" / "serve_interactive_fastapi_demo.py"
    spec = importlib.util.spec_from_file_location("interactive_demo_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    monkeypatch.setattr(
        launcher,
        "PAPER_STANDARD_BUNDLE_PATH",
        tmp_path / "missing-standard-bundle.joblib",
    )
    monkeypatch.setattr(
        launcher,
        "PAPER_SIDECAR_BUNDLE_PATH",
        tmp_path / "missing-sidecar-bundle.joblib",
    )
    monkeypatch.setenv("GALACTICUS_EMU_PROJECT_ROOT", "/stale/project/root")
    monkeypatch.setenv("INTERACTIVE_OBSERVABLES_BUNDLE_PATH", "/stale/bundle/path")
    monkeypatch.setenv(
        "INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH",
        "/stale/summary/path",
    )
    monkeypatch.setenv(
        "INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH",
        "/stale/summary/path",
    )

    launcher.configure_demo_app()

    assert Path(os.environ["GALACTICUS_EMU_PROJECT_ROOT"]) == repository_root
    assert os.environ["INTERACTIVE_SMF_ENABLED"] == "0"
    assert os.environ["INTERACTIVE_HALPHA_ENABLED"] == "0"
    assert os.environ["INTERACTIVE_OBSERVABLES_ENABLED"] == "1"
    assert os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] == "0"
    assert "INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH" not in os.environ
    assert "INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH" not in os.environ
    assert "INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS" not in os.environ
    assert "INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS" not in os.environ
    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"]) == (
        launcher.CHECKED_IN_STANDARD_BUNDLE_PATH
    )
    assert launcher.CHECKED_IN_STANDARD_BUNDLE_PATH.is_file()

    expected_html_assets = [
        "assets/interactive_smf_demo/index.html",
        "assets/interactive_halpha_demo/index.html",
        "assets/interactive_observables_demo/index.html",
    ]

    for relative_path in expected_html_assets:
        assert (repository_root / relative_path).is_file()


def test_local_demo_configures_paper_best_fits(monkeypatch, tmp_path) -> None:
    _track_app_environment(monkeypatch)
    repository_root = Path(__file__).resolve().parents[1]
    launcher_path = repository_root / "scripts" / "serve_interactive_fastapi_demo.py"
    spec = importlib.util.spec_from_file_location("interactive_demo_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    paths = {
        "PAPER_STANDARD_BUNDLE_PATH": tmp_path / "standard.joblib",
        "PAPER_SIDECAR_BUNDLE_PATH": tmp_path / "sidecar.joblib",
        "PAPER_STANDARD_BEST_FIT_SUMMARY_PATH": tmp_path / "standard.json",
        "PAPER_SIDECAR_BEST_FIT_SUMMARY_PATH": tmp_path / "sidecar.json",
    }
    for name, path in paths.items():
        path.touch()
        monkeypatch.setattr(launcher, name, path)

    launcher.configure_demo_app()

    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"]) == paths[
        "PAPER_STANDARD_BUNDLE_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH"]) == paths[
        "PAPER_STANDARD_BEST_FIT_SUMMARY_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_SIDECAR_LF_BUNDLE_PATH"]) == paths[
        "PAPER_SIDECAR_BUNDLE_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH"]) == paths[
        "PAPER_SIDECAR_BEST_FIT_SUMMARY_PATH"
    ]
    assert os.environ["INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS"] == "64"
    assert os.environ["INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS"] == "64"
