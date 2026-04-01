from __future__ import annotations

import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ReviewHandler(SimpleHTTPRequestHandler):
    data_dir: Path

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/manifest":
            self._serve_json(self.data_dir / "manifest.json")
        elif self.path.startswith("/api/detection/"):
            name = self.path[len("/api/detection/"):]
            stem = Path(name).stem
            self._serve_json(self.data_dir / "detections" / f"{stem}.json")
        elif self.path.startswith("/images/"):
            name = self.path[len("/images/"):]
            self._serve_file(self.data_dir / "images" / name)
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        html_path = Path(__file__).parent / "review.html"
        self._serve_file(html_path, content_type="text/html")

    def _serve_json(self, path: Path) -> None:
        self._serve_file(path, content_type="application/json")

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        if content_type is None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        # Quiet logging — only errors
        pass


def run_server(data_dir: Path, port: int = 8080) -> None:
    ReviewHandler.data_dir = data_dir.resolve()
    server = HTTPServer(("0.0.0.0", port), ReviewHandler)
    print(f"Review server running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
