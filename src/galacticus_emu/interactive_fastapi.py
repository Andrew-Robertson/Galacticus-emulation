from __future__ import annotations

from functools import lru_cache
import html
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .interactive_halpha import bundle_meta as halpha_bundle_meta
from .interactive_halpha import load_halpha_bundle, predict_halpha_bundle
from .interactive_smf import bundle_meta as smf_bundle_meta
from .interactive_smf import load_smf_bundle, predict_smf_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMF_BUNDLE_PATH = REPO_ROOT / "playing" / "interactive_smf_demo" / "smf_demo_bundle.joblib"
DEFAULT_SMF_HTML_PATH = REPO_ROOT / "assets" / "interactive_smf_demo" / "index.html"
DEFAULT_HALPHA_BUNDLE_PATH = REPO_ROOT / "playing" / "interactive_halpha_demo" / "halpha_demo_bundle_8draws.joblib"
DEFAULT_HALPHA_HTML_PATH = REPO_ROOT / "assets" / "interactive_halpha_demo" / "index.html"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


def _inject_api_base(html_text: str, api_base: str) -> str:
    script = f'  <script>window.API_BASE = "{api_base}";</script>\n'
    if "</head>" in html_text:
        return html_text.replace("</head>", f"{script}</head>", 1)
    return script + html_text


def _bundle_enabled(path: Path) -> bool:
    return path.exists()


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
def get_smf_html() -> str:
    return _inject_api_base(smf_html_path().read_text(), "/smf")


@lru_cache(maxsize=1)
def get_halpha_html() -> str:
    return _inject_api_base(halpha_html_path().read_text(), "/halpha")


def _enabled_demos() -> dict[str, bool]:
    return {
        "smf": _bundle_enabled(smf_bundle_path()),
        "halpha": _bundle_enabled(halpha_bundle_path()),
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
    if not cards:
        cards.append(
            """
            <div class="card disabled">
              <h2>No demo bundles found</h2>
              <p>Set <code>INTERACTIVE_SMF_BUNDLE_PATH</code> or <code>INTERACTIVE_HALPHA_BUNDLE_PATH</code> to a saved bundle before starting the app.</p>
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
      Small interactive viewers for pre-trained Galacticus emulators. These are designed to be cheap
      to host: training happens offline, and the web app only loads a saved bundle and evaluates it live.
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
        }

    @app.get("/smf")
    @app.get("/smf/", include_in_schema=False)
    def smf_page():
        if not _bundle_enabled(smf_bundle_path()):
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
        if not _bundle_enabled(halpha_bundle_path()):
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
