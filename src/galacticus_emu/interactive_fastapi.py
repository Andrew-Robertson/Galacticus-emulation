from __future__ import annotations

from functools import lru_cache
import html
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .interactive_halpha import bundle_meta as halpha_bundle_meta
from .interactive_halpha import load_halpha_bundle, predict_halpha_bundle
from .interactive_observables import bundle_meta as observables_bundle_meta
from .interactive_observables import load_observables_bundle, predict_observables_bundle
from .interactive_observables import refresh_observables_training_preview
from .interactive_sidecar_lf import bundle_meta as sidecar_lf_bundle_meta
from .interactive_sidecar_lf import load_sidecar_lf_bundle, predict_sidecar_lf_bundle
from .interactive_smf import bundle_meta as smf_bundle_meta
from .interactive_smf import load_smf_bundle, predict_smf_bundle


def _default_project_root() -> Path:
    candidates = []
    env_root = os.environ.get("GALACTICUS_EMU_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "assets").exists() and (resolved / "pyproject.toml").exists():
            return resolved
    return Path.cwd().resolve()


PROJECT_ROOT = _default_project_root()


def _first_available_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


DEFAULT_EXAMPLE_CAMPAIGN_ROOT = (
    PROJECT_ROOT / "runs" / "campaigns" / "sobol_1024_20p_simpleSizes_moreHalos_reduced"
)
DEFAULT_EXAMPLE_EMULATOR_ROOT = DEFAULT_EXAMPLE_CAMPAIGN_ROOT / "pipeline" / "emulators"
DEFAULT_SMF_BUNDLE_PATH = PROJECT_ROOT / "demo_emulators" / "interactive_smf_demo" / "smf_demo_bundle.joblib"
DEFAULT_SMF_HTML_PATH = PROJECT_ROOT / "assets" / "interactive_smf_demo" / "index.html"
DEFAULT_HALPHA_BUNDLE_PATH = (
    PROJECT_ROOT / "demo_emulators" / "interactive_halpha_demo" / "halpha_demo_bundle_8draws.joblib"
)
DEFAULT_HALPHA_HTML_PATH = PROJECT_ROOT / "assets" / "interactive_halpha_demo" / "index.html"
DEFAULT_DEMO_OBSERVABLES_BUNDLE_PATH = (
    PROJECT_ROOT / "demo_emulators" / "interactive_observables_demo" / "observables_demo_bundle_pca.joblib"
)
DEFAULT_DEPLOYMENT_OBSERVABLES_BUNDLE_PATH = (
    PROJECT_ROOT / "demo_emulators" / "paper" / "standard_observables_mean_only.joblib"
)
DEFAULT_EXAMPLE_OBSERVABLES_BUNDLE_PATH = (
    DEFAULT_EXAMPLE_EMULATOR_ROOT
    / "standard_observables"
    / "pca_99"
    / "standard_observables_bundle.joblib"
)
DEFAULT_OBSERVABLES_BUNDLE_PATH = _first_available_path(
    DEFAULT_EXAMPLE_OBSERVABLES_BUNDLE_PATH,
    DEFAULT_DEPLOYMENT_OBSERVABLES_BUNDLE_PATH,
    DEFAULT_DEMO_OBSERVABLES_BUNDLE_PATH,
)
DEFAULT_OBSERVABLES_HTML_PATH = PROJECT_ROOT / "assets" / "interactive_observables_demo" / "index.html"
DEFAULT_DEMO_SIDECAR_LF_BUNDLE_PATH = (
    PROJECT_ROOT / "demo_emulators" / "interactive_sidecar_lf_demo" / "sidecar_lf_demo_bundle.joblib"
)
DEFAULT_DEPLOYMENT_SIDECAR_LF_BUNDLE_PATH = (
    PROJECT_ROOT / "demo_emulators" / "paper" / "halpha_sobral_log_error_mean_only.joblib"
)
DEFAULT_EXAMPLE_SIDECAR_LF_BUNDLE_PATH = (
    DEFAULT_EXAMPLE_EMULATOR_ROOT
    / "emission_line_lfs"
    / "pca_99"
    / "halpha_sobral_1dustdraw_pca99.joblib"
)
DEFAULT_SIDECAR_LF_BUNDLE_PATH = _first_available_path(
    DEFAULT_EXAMPLE_SIDECAR_LF_BUNDLE_PATH,
    DEFAULT_DEPLOYMENT_SIDECAR_LF_BUNDLE_PATH,
    DEFAULT_DEMO_SIDECAR_LF_BUNDLE_PATH,
)
DEFAULT_SIDECAR_LF_HTML_PATH = DEFAULT_OBSERVABLES_HTML_PATH
DEFAULT_OBSERVABLES_ORDER = [
    "smf_liwhite2009_sdss",
    "sfr_function_robotham2011",
    "mzr_blanc2019",
    "bh_velocity_dispersion",
    "size_mass_vdw2014_star_forming_z0",
    "size_mass_vdw2014_quiescent_z0",
    "smf_z0",
    "smf_z3",
]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        configured = Path(value).expanduser().resolve()
        if configured.exists() or not (os.environ.get("RENDER") and default.exists()):
            return configured
    return default.resolve()


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _inject_api_base(html_text: str, api_base: str) -> str:
    script = f'  <script>window.API_BASE = "{api_base}";</script>\n'
    if "</head>" in html_text:
        return html_text.replace("</head>", f"{script}</head>", 1)
    return script + html_text


def _bundle_enabled(flag_name: str, path: Path) -> bool:
    default_enabled = flag_name not in {
        "INTERACTIVE_SMF_ENABLED",
        "INTERACTIVE_HALPHA_ENABLED",
    }
    return _env_flag(flag_name, default=default_enabled) and path.exists()


@lru_cache(maxsize=1)
def smf_bundle_path() -> Path:
    return _env_path("INTERACTIVE_SMF_BUNDLE_PATH", DEFAULT_SMF_BUNDLE_PATH)


@lru_cache(maxsize=1)
def smf_html_path() -> Path:
    return _env_path("INTERACTIVE_SMF_HTML_PATH", DEFAULT_SMF_HTML_PATH)


@lru_cache(maxsize=1)
def halpha_bundle_path() -> Path:
    return _env_path("INTERACTIVE_HALPHA_BUNDLE_PATH", DEFAULT_HALPHA_BUNDLE_PATH)


@lru_cache(maxsize=1)
def halpha_html_path() -> Path:
    return _env_path("INTERACTIVE_HALPHA_HTML_PATH", DEFAULT_HALPHA_HTML_PATH)


@lru_cache(maxsize=1)
def observables_bundle_path() -> Path:
    return _env_path("INTERACTIVE_OBSERVABLES_BUNDLE_PATH", DEFAULT_OBSERVABLES_BUNDLE_PATH)


@lru_cache(maxsize=1)
def observables_html_path() -> Path:
    return _env_path("INTERACTIVE_OBSERVABLES_HTML_PATH", DEFAULT_OBSERVABLES_HTML_PATH)


@lru_cache(maxsize=1)
def sidecar_lf_bundle_path() -> Path:
    return _env_path("INTERACTIVE_SIDECAR_LF_BUNDLE_PATH", DEFAULT_SIDECAR_LF_BUNDLE_PATH)


@lru_cache(maxsize=1)
def sidecar_lf_html_path() -> Path:
    return _env_path("INTERACTIVE_SIDECAR_LF_HTML_PATH", DEFAULT_SIDECAR_LF_HTML_PATH)


@lru_cache(maxsize=1)
def observables_hdf5_filename() -> str | None:
    return os.environ.get("INTERACTIVE_OBSERVABLES_HDF5_FILENAME")


@lru_cache(maxsize=1)
def observables_training_preview_rows() -> str | None:
    return os.environ.get("INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS")


@lru_cache(maxsize=1)
def observables_order() -> list[str]:
    value = os.environ.get("INTERACTIVE_OBSERVABLES_ORDER")
    if not value:
        return list(DEFAULT_OBSERVABLES_ORDER)
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def interactive_campaign_root() -> Path | None:
    value = os.environ.get("INTERACTIVE_CAMPAIGN_ROOT")
    if not value:
        return None
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def observables_best_fit_summary_path() -> Path | None:
    value = os.environ.get("INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH")
    if not value:
        return None
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def observables_default_params_path() -> Path | None:
    value = os.environ.get("INTERACTIVE_OBSERVABLES_DEFAULT_PARAMS_PATH")
    if not value:
        return None
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def sidecar_lf_training_preview_rows() -> str | None:
    return os.environ.get("INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS")


@lru_cache(maxsize=1)
def sidecar_lf_best_fit_summary_path() -> Path | None:
    value = os.environ.get("INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH")
    if not value:
        return None
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def get_smf_bundle() -> dict:
    path = smf_bundle_path()
    if not path.exists():
        raise FileNotFoundError(path)
    return load_smf_bundle(path)


@lru_cache(maxsize=1)
def get_halpha_bundle() -> dict:
    path = halpha_bundle_path()
    if not path.exists():
        raise FileNotFoundError(path)
    return load_halpha_bundle(path)


@lru_cache(maxsize=1)
def get_observables_bundle() -> dict:
    path = observables_bundle_path()
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = load_observables_bundle(path)
    if interactive_campaign_root() is not None:
        bundle["campaign_root"] = str(interactive_campaign_root())
    ordered_keys = [
        key
        for key in observables_order()
        if key in bundle["observable_keys"]
    ]
    ordered_keys.extend(key for key in bundle["observable_keys"] if key not in ordered_keys)
    bundle["observable_keys"] = ordered_keys
    preview_rows = observables_training_preview_rows()
    if preview_rows:
        bundle = refresh_observables_training_preview(
            bundle,
            training_preview_rows=preview_rows,
            hdf5_filename=observables_hdf5_filename(),
            use_training_targets=True,
        )
    return bundle


@lru_cache(maxsize=1)
def get_sidecar_lf_bundle() -> dict:
    path = sidecar_lf_bundle_path()
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = load_sidecar_lf_bundle(path)
    if interactive_campaign_root() is not None:
        bundle["campaign_root"] = str(interactive_campaign_root())
    return bundle


@lru_cache(maxsize=1)
def get_smf_html() -> str:
    return _inject_api_base(smf_html_path().read_text(), "/smf")


@lru_cache(maxsize=1)
def get_halpha_html() -> str:
    return _inject_api_base(halpha_html_path().read_text(), "/halpha")


@lru_cache(maxsize=1)
def get_observables_html() -> str:
    return _inject_api_base(observables_html_path().read_text(), "/observables")


@lru_cache(maxsize=1)
def get_sidecar_lf_html() -> str:
    return _inject_api_base(sidecar_lf_html_path().read_text(), "/sidecar-lfs")


def _enabled_demos() -> dict[str, bool]:
    return {
        "smf": _bundle_enabled("INTERACTIVE_SMF_ENABLED", smf_bundle_path()),
        "halpha": _bundle_enabled("INTERACTIVE_HALPHA_ENABLED", halpha_bundle_path()),
        "observables": _bundle_enabled("INTERACTIVE_OBSERVABLES_ENABLED", observables_bundle_path()),
        "sidecar_lfs": _bundle_enabled("INTERACTIVE_SIDECAR_LF_ENABLED", sidecar_lf_bundle_path()),
    }


def _landing_page() -> str:
    demos = _enabled_demos()
    cards = []
    if demos["smf"]:
        cards.append(
            """
            <a class="card" href="/smf">
              <h2>Interactive SMF Demo</h2>
              <p>One slow Galacticus slider, live stellar mass function prediction, and training curves in the background.</p>
            </a>
            """
        )
    if demos["halpha"]:
        cards.append(
            """
            <a class="card" href="/halpha">
              <h2>Interactive Halpha Demo</h2>
              <p>Six sliders, four Sobral redshifts, and a PCA emulator for the dust-attenuated luminosity functions.</p>
            </a>
            """
        )
    if demos["observables"]:
        cards.append(
            """
            <a class="card" href="/observables">
              <h2>Interactive Observable Suite</h2>
              <p>Galacticus parameter sliders and live predictions for stellar mass functions, size relations, the MZR, and related observables.</p>
            </a>
            """
        )
    if demos["sidecar_lfs"]:
        cards.append(
            """
            <a class="card" href="/sidecar-lfs">
              <h2>Interactive Emission-Line LFs</h2>
              <p>Galacticus and dust-parameter sliders with live PCA-GP predictions for saved sidecar luminosity-function emulators.</p>
            </a>
            """
        )
    if not cards:
        cards.append(
            """
            <div class="card disabled">
              <h2>No demo bundles found</h2>
              <p>Set one of <code>INTERACTIVE_SMF_BUNDLE_PATH</code>, <code>INTERACTIVE_HALPHA_BUNDLE_PATH</code>, <code>INTERACTIVE_OBSERVABLES_BUNDLE_PATH</code>, or <code>INTERACTIVE_SIDECAR_LF_BUNDLE_PATH</code> to a saved bundle before starting the app.</p>
            </div>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Galacticus Emulator Demos</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --ink: #1e2832;
      --muted: #68727a;
      --accent: #9d4f24;
      --border: rgba(30, 40, 50, 0.14);
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8ee 0%, rgba(255, 248, 238, 0) 36%),
        linear-gradient(180deg, #f7f0e7 0%, #f2ede6 100%);
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 40px 24px 56px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    .subhead {{
      margin: 0 0 26px;
      max-width: 820px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .card {{
      display: block;
      text-decoration: none;
      color: inherit;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 22px 22px 20px;
      box-shadow: 0 16px 34px rgba(30, 40, 50, 0.06);
      transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 20px 40px rgba(30, 40, 50, 0.08);
    }}
    .card.disabled {{
      opacity: 0.9;
      cursor: default;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .footer {{
      margin-top: 26px;
      font-size: 13px;
      color: var(--muted);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Galacticus Emulator Demo Hub</h1>
    <p class="subhead">
      Explore how Galacticus observables respond as the model parameters change. The emulators are
      trained offline, and the web app evaluates their saved predictions live.
    </p>
    <section class="grid">
      {"".join(cards)}
    </section>
    <div class="footer">
      Health check: <code>/healthz</code>
    </div>
  </main>
</body>
</html>"""


def create_app() -> FastAPI:
    app = FastAPI(title="Galacticus Interactive Emulator Demos")

    @app.get("/", response_class=HTMLResponse)
    def landing() -> str:
        return _landing_page()

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "enabled_demos": _enabled_demos(),
            "smf_bundle_path": str(smf_bundle_path()),
            "halpha_bundle_path": str(halpha_bundle_path()),
            "observables_bundle_path": str(observables_bundle_path()),
            "sidecar_lf_bundle_path": str(sidecar_lf_bundle_path()),
            "interactive_campaign_root": None
            if interactive_campaign_root() is None
            else str(interactive_campaign_root()),
            "observables_hdf5_filename": observables_hdf5_filename(),
            "observables_training_preview_rows": observables_training_preview_rows(),
            "observables_best_fit_summary_path": None
            if observables_best_fit_summary_path() is None
            else str(observables_best_fit_summary_path()),
            "observables_default_params_path": None
            if observables_default_params_path() is None
            else str(observables_default_params_path()),
            "sidecar_lf_training_preview_rows": sidecar_lf_training_preview_rows(),
            "sidecar_lf_best_fit_summary_path": None
            if sidecar_lf_best_fit_summary_path() is None
            else str(sidecar_lf_best_fit_summary_path()),
        }

    @app.get("/smf")
    @app.get("/smf/", include_in_schema=False)
    def smf_page():
        if not _bundle_enabled("INTERACTIVE_SMF_ENABLED", smf_bundle_path()):
            raise HTTPException(status_code=404, detail="SMF demo bundle not available.")
        return HTMLResponse(get_smf_html())

    @app.get("/smf/api/meta")
    def smf_meta():
        try:
            return JSONResponse(smf_bundle_meta(get_smf_bundle()))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"SMF bundle not found: {exc}") from exc

    @app.get("/smf/api/predict")
    def smf_predict(diskVelocityCharacteristic: float = Query(...)):
        try:
            bundle = get_smf_bundle()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"SMF bundle not found: {exc}") from exc
        params = {"diskVelocityCharacteristic": float(diskVelocityCharacteristic)}
        prediction = predict_smf_bundle(bundle, params)
        return {
            "params": params,
            "y_pred_log10": prediction["y_pred_log10"].tolist(),
            "y_std_log10": prediction["y_std_log10"].tolist(),
        }

    @app.get("/halpha")
    @app.get("/halpha/", include_in_schema=False)
    def halpha_page():
        if not _bundle_enabled("INTERACTIVE_HALPHA_ENABLED", halpha_bundle_path()):
            raise HTTPException(status_code=404, detail="Halpha demo bundle not available.")
        return HTMLResponse(get_halpha_html())

    @app.get("/halpha/api/meta")
    def halpha_meta():
        try:
            return JSONResponse(halpha_bundle_meta(get_halpha_bundle()))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Halpha bundle not found: {exc}") from exc

    @app.get("/halpha/api/predict")
    def halpha_predict(
        diskVelocityCharacteristic: float = Query(...),
        delta_0: float = Query(...),
        delta_z: float = Query(...),
        delta_M: float = Query(...),
        delta_Mz: float = Query(...),
        attenuation_scatter: float = Query(...),
    ):
        try:
            bundle = get_halpha_bundle()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Halpha bundle not found: {exc}") from exc
        params = {
            "diskVelocityCharacteristic": float(diskVelocityCharacteristic),
            "delta_0": float(delta_0),
            "delta_z": float(delta_z),
            "delta_M": float(delta_M),
            "delta_Mz": float(delta_Mz),
            "attenuation_scatter": float(attenuation_scatter),
        }
        prediction = predict_halpha_bundle(bundle, params)
        return {
            "params": params,
            "predictions": {
                sobral_label: {
                    "log10_luminosity_center": payload["log10_luminosity_center"].tolist(),
                    "y_pred_log10": payload["y_pred_log10"].tolist(),
                    "y_std_log10": payload["y_std_log10"].tolist(),
                }
                for sobral_label, payload in prediction.items()
            },
        }

    @app.get("/observables")
    @app.get("/observables/", include_in_schema=False)
    def observables_page():
        if not _bundle_enabled("INTERACTIVE_OBSERVABLES_ENABLED", observables_bundle_path()):
            raise HTTPException(status_code=404, detail="Observables demo bundle not available.")
        return HTMLResponse(get_observables_html())

    @app.get("/observables/api/meta")
    def observables_meta():
        try:
            return JSONResponse(
                observables_bundle_meta(
                    get_observables_bundle(),
                    best_fit_summary_path=observables_best_fit_summary_path(),
                    default_params_path=observables_default_params_path(),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Observables bundle not found: {exc}") from exc

    @app.get("/observables/api/predict")
    def observables_predict(request: Request):
        try:
            bundle = get_observables_bundle()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Observables bundle not found: {exc}") from exc

        params = {}
        for column in bundle["input_columns"]:
            value = request.query_params.get(column)
            if value is None:
                raise HTTPException(status_code=422, detail=f"Missing required query parameter: {column}")
            params[column] = float(value)

        prediction = predict_observables_bundle(bundle, params)
        return {
            "params": params,
            "predictions": {
                observable_key: {
                    "x_plot": payload["x_plot"].tolist(),
                    "y_pred_plot": payload["y_pred_plot"].tolist(),
                    "y_std_plot": payload["y_std_plot"].tolist(),
                }
                for observable_key, payload in prediction.items()
            },
        }

    @app.get("/sidecar-lfs")
    @app.get("/sidecar-lfs/", include_in_schema=False)
    def sidecar_lf_page():
        if not _bundle_enabled("INTERACTIVE_SIDECAR_LF_ENABLED", sidecar_lf_bundle_path()):
            raise HTTPException(status_code=404, detail="Sidecar LF demo bundle not available.")
        return HTMLResponse(get_sidecar_lf_html())

    @app.get("/sidecar-lfs/api/meta")
    def sidecar_lf_meta():
        try:
            return JSONResponse(
                sidecar_lf_bundle_meta(
                    get_sidecar_lf_bundle(),
                    best_fit_summary_path=sidecar_lf_best_fit_summary_path(),
                    training_preview_rows=sidecar_lf_training_preview_rows(),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Sidecar LF bundle not found: {exc}") from exc

    @app.get("/sidecar-lfs/api/predict")
    def sidecar_lf_predict(request: Request):
        try:
            bundle = get_sidecar_lf_bundle()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Sidecar LF bundle not found: {exc}") from exc

        params = {}
        for column in bundle["parameter_names"]:
            value = request.query_params.get(column)
            if value is None:
                raise HTTPException(status_code=422, detail=f"Missing required query parameter: {column}")
            params[column] = float(value)

        prediction = predict_sidecar_lf_bundle(bundle, params)
        return {
            "params": params,
            "predictions": {
                observable_key: {
                    "x_plot": payload["x_plot"].tolist(),
                    "y_pred_plot": payload["y_pred_plot"].tolist(),
                    "y_std_plot": payload["y_std_plot"].tolist(),
                }
                for observable_key, payload in prediction.items()
            },
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        raise HTTPException(status_code=404, detail="No favicon")

    @app.get("/{path:path}", include_in_schema=False)
    def fallback(path: str):
        safe_path = html.escape(path)
        return HTMLResponse(
            f"<h1>Not found</h1><p>No route matched <code>/{safe_path}</code>.</p>",
            status_code=404,
        )

    return app


app = create_app()
