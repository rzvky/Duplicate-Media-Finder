"""Local review screen (runs only on this computer at http://127.0.0.1)."""

from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from . import actions, report
from .loader import load_image
from .paths import lp

BROWSER_SAFE = {"JPEG", "PNG", "GIF", "WEBP", "BMP"}
# Video containers a browser may be able to play (the page falls back to a picture strip otherwise)
VIDEO_TYPES = {
    "MP4": "video/mp4",
    "M4V": "video/mp4",
    "MOV": "video/mp4",
    "3GP": "video/mp4",
    "3G2": "video/mp4",
    "WEBM": "video/webm",
    "MKV": "video/webm",
    "OGV": "video/ogg",
}


class ReviewApp:
    def __init__(self, root: Path, out_dir: Path, results: dict):
        self.root = Path(root)
        self.out_dir = Path(out_dir)
        self.results = results
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.strip_lock = threading.Lock()
        self.by_fid = {m["fid"]: m for g in results["groups"] for m in g["files"]}
        actions.apply_status(self.results, self.root)
        self.results["batches"] = actions.batches(self.out_dir)
        self.httpd: Optional[ThreadingHTTPServer] = None

    # ---- actions
    def move(self, fids) -> dict:
        with self.lock:
            actions.apply_status(self.results, self.root)
            allowed, blocked = actions.guard_selection(self.results, set(fids), self.root)
            res = actions.move_to_bin(self.root, self.out_dir, allowed)
            dest = {x["rel"]: x["dst"] for x in res["moved_to"]}
            moved = set(dest)
            for m in allowed:
                if m["rel"] in moved:
                    m["status"] = "moved"
                    m["selected"] = False
                    m["bin_rel"] = dest[m["rel"]]
            res["blocked_groups"] = blocked
            res["moved_bytes"] = int(sum(m["size"] for m in allowed if m["rel"] in moved))
            self._save()
            return res

    def undo(self, all_batches=False) -> dict:
        with self.lock:
            res = actions.undo(self.root, self.out_dir, all_batches=all_batches)
            restored = set(res["restored"])
            for m in self.by_fid.values():
                if m["rel"] in restored:
                    m["status"] = "present"
                    m["selected"] = True  # it was ticked when it was moved
            actions.apply_status(self.results, self.root)
            self._save()
            return res

    def _save(self):
        self.results["batches"] = actions.batches(self.out_dir)
        try:
            report.save_all(self.out_dir, self.results)
        except OSError:
            pass

    def _path(self, m: dict) -> Path:
        if m.get("status") == "moved":
            return self.out_dir / m["bin_rel"] if m.get("bin_rel") else actions.review_bin(self.out_dir) / m["rel"]
        return self.root / m["rel"]

    def reveal(self, fid: str) -> bool:
        m = self.by_fid.get(fid)
        if not m:
            return False
        path = self._path(m)
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(f'explorer /select,"{path}"')
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
            return True
        except OSError:
            return False

    def video_file(self, fid: str) -> Optional[tuple]:
        m = self.by_fid.get(fid)
        if not m or m.get("media") != "video":
            return None
        path = self._path(m)
        if not os.path.isfile(lp(path)):
            return None
        return path, VIDEO_TYPES.get(m.get("fmt", ""), "application/octet-stream")

    def strip_bytes(self, fid: str) -> Optional[bytes]:
        """Six pictures across a video (for formats the browser cannot play). Made once, then kept."""
        m = self.by_fid.get(fid)
        if not m or m.get("media") != "video":
            return None
        dst = self.out_dir / "thumbs" / f"strip-{m['fid']}-{m.get('sha12', 'x')}.jpg"
        with self.strip_lock:
            if not dst.is_file():
                path = self._path(m)
                if not os.path.isfile(lp(path)):
                    return None
                from .video import make_video_strip

                dst.parent.mkdir(parents=True, exist_ok=True)
                if make_video_strip(str(path), str(dst)) is not None:
                    return None
        return dst.read_bytes()

    def preview_bytes(self, fid: str) -> Optional[tuple]:
        m = self.by_fid.get(fid)
        if not m:
            return None
        if m.get("media") == "video":
            p = self.out_dir / "thumbs" / report.thumb_name(m)
            return (p.read_bytes(), "image/jpeg") if p.is_file() else None
        path = self._path(m)
        if not os.path.isfile(lp(path)):
            return None
        if m["fmt"] in BROWSER_SAFE and not m.get("raw") and m["size"] < 40_000_000:
            ctype = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            with open(lp(path), "rb") as f:
                return f.read(), ctype
        with open(lp(path), "rb") as f:
            data = f.read()
        li = load_image(str(path), data, work_side=2048)
        buf = io.BytesIO()
        li.image.save(buf, "JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"

    # ---- server
    def serve(self, port: int = 8765, open_browser: bool = True) -> str:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "DuplicateMediaFinder"

            def log_message(self, *a):
                pass

            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").split(":")[0].strip("[]").lower()
                return host in ("127.0.0.1", "localhost", "::1")

            timeout = 30  # a stalled request cannot hold a thread forever

            def _token_ok(self) -> bool:
                if self.headers.get("X-Token") == app.token:
                    return True
                q = parse_qs(urlparse(self.path).query)
                return q.get("t", [""])[0] == app.token

            def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _send_file(self, path, ctype: str, head: bool = False):
                """Stream a file, honouring Range requests (needed for seeking in videos)."""
                try:
                    size = os.path.getsize(lp(path))
                except OSError:
                    return self._send(404, b"", "text/plain")
                start, end, code = 0, size - 1, 200
                rng = (self.headers.get("Range") or "").strip()
                m = re.match(r"bytes=(\d*)-(\d*)$", rng) if rng else None
                if m and (m.group(1) or m.group(2)):
                    a, b = m.groups()
                    if not a:
                        start, end = max(0, size - int(b)), size - 1
                    else:
                        start, end = int(a), (int(b) if b else size - 1)
                    end = min(end, size - 1)
                    if start >= size or start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    code = 206
                length = end - start + 1
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                if code == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if head:
                    return
                try:
                    with open(lp(path), "rb") as f:
                        f.seek(start)
                        left = length
                        while left > 0:
                            chunk = f.read(min(1 << 20, left))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _json(self, obj, code=200):
                self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

            def do_GET(self):
                if not self._host_ok():
                    return self._send(403, b"forbidden", "text/plain")
                path = unquote(urlparse(self.path).path)
                if path in ("/", "/index.html"):
                    page = report.render_page(None, live=True, token=app.token)
                    return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                if path == "/api/results":
                    if self.headers.get("X-Token") != app.token:
                        return self._json({"error": "forbidden"}, 403)
                    with app.lock:
                        return self._json(app.results)
                if path.startswith(("/thumb/", "/full/", "/video/", "/strip/")) and not self._token_ok():
                    return self._send(403, b"", "text/plain")
                if path.startswith("/video/"):
                    vf = app.video_file(path[len("/video/") :])
                    if vf is None:
                        return self._send(404, b"", "text/plain")
                    return self._send_file(vf[0], vf[1])
                if path.startswith("/strip/") and path.endswith(".jpg"):
                    try:
                        data = app.strip_bytes(path[len("/strip/") : -4])
                    except Exception:
                        data = None
                    if data is None:
                        return self._send(404, b"", "image/jpeg")
                    return self._send(200, data, "image/jpeg", cache=True)
                if path.startswith("/thumb/") and path.endswith(".jpg"):
                    fid = path[len("/thumb/") : -4]
                    m = app.by_fid.get(fid)
                    if m:
                        p = app.out_dir / "thumbs" / report.thumb_name(m)
                        if p.is_file():
                            return self._send(200, p.read_bytes(), "image/jpeg", cache=True)
                    return self._send(404, b"", "image/jpeg")
                if path.startswith("/full/"):
                    fid = path[len("/full/") :]
                    try:
                        res = app.preview_bytes(fid)
                    except Exception:
                        res = None
                    if res is None:
                        return self._send(404, b"", "text/plain")
                    return self._send(200, res[0], res[1])
                return self._send(404, b"not found", "text/plain")

            def do_HEAD(self):
                """Some players ask for the size first."""
                path = unquote(urlparse(self.path).path)
                if not self._host_ok() or not path.startswith("/video/") or not self._token_ok():
                    self.send_response(403 if path.startswith("/video/") else 405)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                vf = app.video_file(path[len("/video/") :])
                if vf is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                return self._send_file(vf[0], vf[1], head=True)

            def do_POST(self):
                if not self._host_ok() or self.headers.get("X-Token") != app.token:
                    return self._json({"error": "forbidden"}, 403)
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    if n < 0 or n > 20_000_000:
                        return self._json({"error": "request too large"}, 413)
                    body = json.loads(self.rfile.read(n) or b"{}")
                except (ValueError, OSError):
                    return self._json({"error": "bad request"}, 400)
                if not isinstance(body, dict):
                    return self._json({"error": "bad request"}, 400)
                path = urlparse(self.path).path
                try:
                    if path == "/api/move":
                        fids = body.get("fids") or []
                        if not isinstance(fids, list):
                            return self._json({"error": "bad request"}, 400)
                        return self._json(app.move([str(f) for f in fids]))
                    if path == "/api/undo":
                        return self._json(app.undo(all_batches=bool(body.get("all"))))
                    if path == "/api/reveal":
                        return self._json({"ok": app.reveal(str(body.get("fid", "")))})
                    if path == "/api/select":  # remember ticks between sessions
                        sel = body.get("selected") or {}
                        if not isinstance(sel, dict):
                            return self._json({"error": "bad request"}, 400)
                        with app.lock:
                            for fid, val in sel.items():
                                if fid in app.by_fid:
                                    app.by_fid[fid]["selected"] = bool(val)
                        return self._json({"ok": True})
                    if path == "/api/quit":
                        self._json({"ok": True})
                        threading.Thread(target=app.httpd.shutdown, daemon=True).start()
                        return
                except Exception as exc:  # report, never crash the server
                    return self._json({"error": str(exc)[:300]}, 500)
                return self._json({"error": "not found"}, 404)

        httpd = None
        for p in range(port, port + 40):
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                break
            except OSError:
                continue
        if httpd is None:
            raise OSError("No free local port found for the review screen")
        httpd.daemon_threads = True
        self.httpd = httpd
        url = f"http://127.0.0.1:{httpd.server_address[1]}/"
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        return url

    def run_forever(self):
        try:
            self.httpd.serve_forever(poll_interval=0.3)
        finally:
            with self.lock:
                self._save()
            self.httpd.server_close()
