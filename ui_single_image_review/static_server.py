"""
Simple static file server for live development of the UI.
Serves HTML/CSS/JS from ui_single_image_review/ directory on port 8000.
Still connects to inference server on port 8765 for /detect requests.

Usage:
    python ui_single_image_review/static_server.py

Access:
    http://127.0.0.1:8000/
"""

import http.server
import socketserver
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
PORT = 8000

class CacheBypassHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self):
        # Add no-cache headers to force refresh
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[static] {self.address_string()} - {format % args}", flush=True)


if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), CacheBypassHandler) as httpd:
        print(f"[static] Static file server starting...")
        print(f"[static] Source  : {UI_DIR}")
        print(f"[static] Open    : http://127.0.0.1:{PORT}/")
        print(f"[static] (Inference still on port 8765)")
        print(f"[static] Press Ctrl+C to stop.", flush=True)
        httpd.serve_forever()
