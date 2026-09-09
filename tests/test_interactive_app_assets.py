from __future__ import annotations

import importlib.util
import os
from pathlib import Path

APP_ENVIRONMENT_VARIABLES = [
    "GALACTICUS_EMU_PROJECT_ROOT",
    "INTERACTIVE_SMF_ENABLED",
    "INTERACTIVE_HALPHA_ENABLED",
    "INTERACTIVE_CAMPAIGN_ROOT",
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


def test_hosted_demo_uses_compact_paper_bundles(monkeypatch, tmp_path) -> None:
    _track_app_environment(monkeypatch)
    repository_root = Path(__file__).resolve().parents[1]
    launcher_path = repository_root / "scripts" / "serve_interactive_fastapi_demo.py"
    spec = importlib.util.spec_from_file_location("interactive_demo_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    standard_bundle = tmp_path / "standard-mean-only.joblib"
    sidecar_bundle = tmp_path / "sidecar-mean-only.joblib"
    standard_bundle.touch()
    sidecar_bundle.touch()
    monkeypatch.setattr(launcher, "DEPLOYMENT_STANDARD_BUNDLE_PATH", standard_bundle)
    monkeypatch.setattr(launcher, "DEPLOYMENT_SIDECAR_BUNDLE_PATH", sidecar_bundle)
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

    launcher.configure_demo_app(deployment=True)

    assert Path(os.environ["GALACTICUS_EMU_PROJECT_ROOT"]) == repository_root
    assert os.environ["INTERACTIVE_SMF_ENABLED"] == "0"
    assert os.environ["INTERACTIVE_HALPHA_ENABLED"] == "0"
    assert os.environ["INTERACTIVE_OBSERVABLES_ENABLED"] == "1"
    assert os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] == "1"
    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH"]) == (
        launcher.PAPER_BEST_FIT_SUMMARY_PATH
    )
    assert Path(os.environ["INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH"]) == (
        launcher.PAPER_BEST_FIT_SUMMARY_PATH
    )
    assert "INTERACTIVE_CAMPAIGN_ROOT" not in os.environ
    assert "INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS" not in os.environ
    assert "INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS" not in os.environ
    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"]) == (
        standard_bundle
    )
    assert Path(os.environ["INTERACTIVE_SIDECAR_LF_BUNDLE_PATH"]) == sidecar_bundle

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
        "PAPER_BEST_FIT_SUMMARY_PATH": tmp_path / "joint.json",
    }
    for name, path in paths.items():
        path.touch()
        monkeypatch.setattr(launcher, name, path)

    launcher.configure_demo_app()

    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"]) == paths[
        "PAPER_STANDARD_BUNDLE_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH"]) == paths[
        "PAPER_BEST_FIT_SUMMARY_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_SIDECAR_LF_BUNDLE_PATH"]) == paths[
        "PAPER_SIDECAR_BUNDLE_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH"]) == paths[
        "PAPER_BEST_FIT_SUMMARY_PATH"
    ]
    assert Path(os.environ["INTERACTIVE_CAMPAIGN_ROOT"]) == launcher.PAPER_CAMPAIGN_ROOT
    assert os.environ["INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS"] == "64"
    assert os.environ["INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS"] == "64"


def test_observable_display_order_matches_paper() -> None:
    from galacticus_emu.interactive_fastapi import DEFAULT_OBSERVABLES_ORDER

    assert DEFAULT_OBSERVABLES_ORDER == [
        "smf_z0",
        "smf_z3",
        "size_mass_vdw2014_star_forming_z0",
        "size_mass_vdw2014_quiescent_z0",
        "sfr_function_robotham2011",
        "bh_velocity_dispersion",
        "mzr_blanc2019",
    ]


def test_interactive_pages_have_route_specific_public_copy() -> None:
    from galacticus_emu.interactive_fastapi import (
        get_observables_html,
        get_sidecar_lf_html,
    )

    get_observables_html.cache_clear()
    get_sidecar_lf_html.cache_clear()
    observables_html = get_observables_html()
    sidecar_lf_html = get_sidecar_lf_html()

    assert 'window.API_BASE = "/observables"' in observables_html
    assert 'window.PAGE_TITLE = "Interactive Multi-Observable Galacticus Demo"' in observables_html
    assert "predicted galaxy observables respond" in observables_html
    assert 'window.API_BASE = "/sidecar-lfs"' in sidecar_lf_html
    assert 'window.PAGE_TITLE = "Interactive H-alpha Luminosity-Function Demo"' in sidecar_lf_html
    assert "predicted H-alpha luminosity functions" in sidecar_lf_html
    assert "target comparisons" not in observables_html
    assert "target comparisons" not in sidecar_lf_html


def test_interactive_legend_precedes_plot_grid() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    html_text = (
        repository_root / "assets" / "interactive_observables_demo" / "index.html"
    ).read_text()

    assert html_text.index('<div class="legend">') < html_text.index(
        '<div class="plot-grid" id="plot-grid"></div>'
    )


def test_landing_page_uses_public_facing_demo_descriptions(monkeypatch) -> None:
    from galacticus_emu import interactive_fastapi

    monkeypatch.setattr(
        interactive_fastapi,
        "_enabled_demos",
        lambda: {
            "smf": False,
            "halpha": False,
            "observables": True,
            "sidecar_lfs": True,
        },
    )

    html_text = interactive_fastapi._landing_page()

    assert "Galacticus Emulator Demos" in html_text
    assert "Galaxy Observables" in html_text
    assert "H-alpha Luminosity Functions" in html_text
    assert "observational measurements" in html_text
    assert "sidecar luminosity-function emulators" not in html_text.lower()
    assert "Health check" not in html_text
