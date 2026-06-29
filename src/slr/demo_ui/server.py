"""Small HTTP API for the React demo UI."""

from __future__ import annotations

import argparse
import cgi
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from slr.demo_ui.inference_service import DemoInferenceService
from slr.demo_ui import settings


class DemoUIRequestHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for the local React demo UI."""

    server_version = "SLRDemoUI/1.0"

    @property
    def app(self) -> DemoInferenceService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return super().log_message(format, *args)

    def _send_json(self, payload: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"ok": True})
            return
        if self.path == "/api/settings":
            self._send_json(self.app.describe())
            return
        self._send_json(
            {
                "ok": False,
                "error": "Not found.",
                "hint": "Use /api/health, /api/settings, or POST /api/predict.",
            },
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json({"ok": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            if "video" not in form:
                raise ValueError("Missing required file field 'video'.")
            video_field = form["video"]
            if getattr(video_field, "file", None) is None:
                raise ValueError("Uploaded 'video' field is empty.")
            branch = form.getfirst("branch", "skeleton")
            payload = self.app.predict(
                branch=str(branch),
                filename=str(getattr(video_field, "filename", "") or "upload.mp4"),
                content=video_field.file.read(),
            )
            self._send_json({"ok": True, "result": payload})
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                },
                status=HTTPStatus.BAD_REQUEST,
            )


class DemoUIServer(ThreadingHTTPServer):
    """HTTP server that carries the shared inference service."""

    def __init__(self, server_address: tuple[str, int], service: DemoInferenceService) -> None:
        super().__init__(server_address, DemoUIRequestHandler)
        self.service = service


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the demo UI backend."""

    parser = argparse.ArgumentParser(description="Run the React demo UI backend.")
    parser.add_argument("--host", type=str, default=settings.API_HOST, help="Bind host.")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Bind port.")
    return parser


def main() -> int:
    """Run the local backend server."""

    parser = build_parser()
    args = parser.parse_args()
    service = DemoInferenceService()
    server = DemoUIServer((str(args.host), int(args.port)), service)
    print(f"Demo UI backend listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
