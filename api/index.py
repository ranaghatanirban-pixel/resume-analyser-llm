# Vercel serverless entry-point for the ResumeAI Streamlit app.
# This module is loaded by Vercel's Python runtime; it wraps
# `streamlit run` via a lightweight ASGI/WSGI bridge so the
# Streamlit server starts on demand.
#
# NOTE: Streamlit apps are stateful long-running servers. Vercel's
# serverless functions are ephemeral, so this bridge starts a
# subprocess Streamlit instance and reverse-proxies to it.  For
# production use we recommend Streamlit Community Cloud, Hugging
# Face Spaces, or a persistent PaaS (Railway, Render, Fly.io)
# instead of Vercel serverless.

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
_STARTED = threading.Event()
_PROC: subprocess.Popen | None = None  # type: ignore[type-arg]


def _start_streamlit() -> None:
    global _PROC
    if _PROC is not None:
        return
    env = {**os.environ, "HF_TOKEN": os.getenv("HF_TOKEN", "")}
    _PROC = subprocess.Popen(  # noqa: S603
        [
            sys.executable, "-m", "streamlit", "run",
            str(APP_DIR / "app.py"),
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true",
            "--server.address", "127.0.0.1",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ],
        env=env,
        cwd=str(APP_DIR),
    )
    # Wait until Streamlit is ready
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health")
            _STARTED.set()
            return
        except Exception:
            time.sleep(1)


class _ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def _proxy(self) -> None:
        _start_streamlit()
        _STARTED.wait(timeout=90)
        target = f"http://127.0.0.1:{STREAMLIT_PORT}{self.path}"
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(target, data=body, method=self.command)
        for k, v in self.headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as exc:
            self.send_error(502, str(exc))

    def log_message(self, *args: object) -> None:  # silence logs
        pass


def handler(request: object, response: object) -> None:  # Vercel signature
    """Vercel calls this for every HTTP request."""
    _start_streamlit()


# Local dev: python api/index.py
if __name__ == "__main__":
    _start_streamlit()
    _STARTED.wait()
    server = HTTPServer(("0.0.0.0", 3000), _ProxyHandler)  # noqa: S104
    print("Proxy listening on http://localhost:3000")
    server.serve_forever()
