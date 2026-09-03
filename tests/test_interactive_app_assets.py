from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def test_checked_in_demo_assets_are_discoverable(monkeypatch) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher_path = repository_root / "scripts" / "serve_interactive_fastapi_demo.py"
    spec = importlib.util.spec_from_file_location("interactive_demo_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    monkeypatch.setenv("GALACTICUS_EMU_PROJECT_ROOT", "/stale/project/root")
    for variable in launcher.CHECKED_IN_DEMOS:
        monkeypatch.setenv(variable, "/stale/bundle/path")

    launcher.configure_checked_in_demos()

    assert Path(os.environ["GALACTICUS_EMU_PROJECT_ROOT"]) == repository_root
    for variable, relative_path in launcher.CHECKED_IN_DEMOS.items():
        expected_path = repository_root / relative_path
        assert expected_path.is_file()
        assert Path(os.environ[variable]) == expected_path

    expected_html_assets = [
        "assets/interactive_smf_demo/index.html",
        "assets/interactive_halpha_demo/index.html",
        "assets/interactive_observables_demo/index.html",
    ]

    for relative_path in expected_html_assets:
        assert (repository_root / relative_path).is_file()
