from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu.interactive_halpha import bundle_meta, load_halpha_bundle, predict_halpha_bundle


DEFAULT_BUNDLE_PATH = REPO_ROOT / "demo_emulators" / "interactive_halpha_demo" / "halpha_demo_bundle_8draws.joblib"
DEFAULT_HTML_PATH = REPO_ROOT / "assets" / "interactive_halpha_demo" / "index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive Halpha LF slider demo locally.")
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_halpha_bundle(args.bundle_path)
    meta = bundle_meta(bundle)
    html = args.html_path.read_text()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = payload.encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[interactive-halpha-demo] {self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(html)
                return
            if parsed.path == "/api/meta":
                self._send_json(meta)
                return
            if parsed.path == "/api/predict":
                query = parse_qs(parsed.query)
                params = {}
                for column in bundle["input_columns"]:
                    default = bundle["input_ranges"][column]["default"]
                    params[column] = float(query.get(column, [default])[0])
                prediction = predict_halpha_bundle(bundle, params)
                response = {
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
                self._send_json(response)
                return
            self.send_error(HTTPStatus.NOT_FOUND.value, "Not found")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving interactive Halpha demo at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
