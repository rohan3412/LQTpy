"""A small web front end: upload a mask, get the report
(``lqtpy serve``).

For colleagues without Python: one page with a file field and a few
options; the report comes back as a self-contained HTML page they can save
or print. Standard library only (``http.server``); every request runs the
normal fast core in-process.

Privacy and safety: it listens on ``127.0.0.1`` (this computer only)
unless ``--host`` says otherwise; uploads are written to a private temp
folder, analysed, and deleted straight after (the report is sent back and
not kept). Anyone who can reach the port can upload a file, so only open
it to a network you trust. Uploads are capped at ``max_mb`` (default 300).
"""
from __future__ import annotations

import html
import os
import shutil
import tempfile
import threading
from email.parser import BytesParser
from email.policy import default as _email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["serve", "make_server"]

_FORM = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LQTpy</title>
<style>
:root{color-scheme:light dark;--bg:#f4f5f7;--panel:#fff;--ink:#121820;--ink2:#5f6877;--line:#d6dbe2;--acc:#d9582a}
@media (prefers-color-scheme:dark){:root{--bg:#0d1015;--panel:#151a21;--ink:#e9edf2;--ink2:#8f98a6;--line:#2a313b;--acc:#f07a45}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:640px;margin:8vh auto;padding:0 20px}
h1{font-size:26px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 24px}
form{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;display:grid;gap:16px}
label{display:grid;gap:6px;font-weight:600;font-size:13px}
input,select{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink)}
.row{display:flex;gap:16px;flex-wrap:wrap}.row label{flex:1;min-width:180px}
.check{display:flex;gap:8px;align-items:center;font-weight:500}
button{font:inherit;font-weight:650;padding:10px 16px;border:0;border-radius:10px;background:var(--acc);color:#fff;cursor:pointer}
.note{color:var(--ink2);font-size:13px}.err{color:#c0342f;font-weight:600}
</style></head><body><main>
<h1>LQTpy</h1><p class="sub">Upload a lesion mask in MNI space (.nii, .nii.gz, .mgz, .nrrd).
You get the full report back: a dashboard, a clinical view and an Export PDF button.</p>
{error}
<form method="post" action="/analyse" enctype="multipart/form-data">
<label>Lesion mask<input type="file" name="lesion" required accept=".nii,.gz,.mgz,.mgh,.nrrd,.nhdr"></label>
<div class="row"><label>Description style<select name="profile">
<option value="stroke">Stroke</option><option value="small_vessel">Small vessel / MS</option>
<option value="tumour">Tumour / resection</option><option value="research">Research</option></select></label>
<label>Expected side<select name="expected_side"><option value="">Not given</option>
<option value="left">Left</option><option value="right">Right</option></select></label></div>
<label class="check"><input type="checkbox" name="extras" value="all"> Add extra measures (shape)</label>
<button type="submit">Analyse</button>
<p class="note">The file is deleted as soon as the report is made; nothing is kept on this computer.
This report is not a clinical diagnosis.</p>
</form></main></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "lqtpy"
    max_bytes = 300 * 1024 * 1024

    def log_message(self, fmt, *args):  # quiet; the CLI prints a line per report
        pass

    def _send(self, code, body: str, ctype="text/html; charset=utf-8", extra=None):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _form(self, error=""):
        msg = f'<p class="err">{html.escape(error)}</p>' if error else ""
        return _FORM.replace("{error}", msg)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, self._form())
        elif self.path == "/health":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, self._form("Not found."))

    def do_POST(self):
        if self.path != "/analyse":
            return self._send(404, self._form("Not found."))
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > self.max_bytes:
            return self._send(413, self._form(f"The upload must be under {self.max_bytes // 2**20} MB."))
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._send(400, self._form("Send the form, please."))
        raw = self.rfile.read(n)
        msg = BytesParser(policy=_email_policy).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
        fields, upload = {}, None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if part.get_filename() is not None:
                upload = (part.get_filename(), part.get_payload(decode=True) or b"")
            elif name:
                fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace").strip()
        if not upload or not upload[1]:
            return self._send(400, self._form("Choose a lesion file first."))
        from .formats import is_lesion_file
        fname = os.path.basename(upload[0].replace("\\", "/")) or "lesion.nii.gz"
        if not is_lesion_file(fname):
            return self._send(400, self._form("That doesn't look like a NIfTI, MGZ or NRRD file."))
        tmp = tempfile.mkdtemp(prefix="lqtpy-web-")
        try:
            path = os.path.join(tmp, fname)
            with open(path, "wb") as fh:
                fh.write(upload[1])
            from .analysis import analyze_single
            profile = fields.get("profile") if fields.get("profile") in (
                "stroke", "small_vessel", "tumour", "research") else "stroke"
            side = fields.get("expected_side") if fields.get("expected_side") in ("left", "right") else None
            res = analyze_single(path, out_dir=os.path.join(tmp, "out"), plots=True, profile=profile,
                                 expected_side=side, extras="all" if fields.get("extras") == "all" else None,
                                 anonymise=os.path.join(tmp, "key.csv"))
            with open(res["report_path"], encoding="utf-8") as fh:
                page = fh.read()
            print(f"  report for an upload of {len(upload[1]) // 1024:,} KB ({res['elapsed']})", flush=True)
            self._send(200, page, extra={"Content-Disposition": 'inline; filename="lqtpy_report.html"'})
        except Exception as exc:
            self._send(400, self._form(f"Couldn't analyse that file: {str(exc).splitlines()[0]}"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def make_server(host: str = "127.0.0.1", port: int = 8765, *, max_mb: int = 300) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"max_bytes": int(max_mb) * 1024 * 1024})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def serve(host: str = "127.0.0.1", port: int = 8765, *, max_mb: int = 300, open_browser: bool = False) -> None:
    """Run the upload page until Ctrl+C."""
    from . import _term
    from .atlases import preload
    srv = make_server(host, port, max_mb=max_mb)
    url = f"http://{'localhost' if host in ('127.0.0.1', '::1') else host}:{srv.server_address[1]}/"
    _term.header("web front end")
    _term.field("Open", url)
    if host not in ("127.0.0.1", "localhost", "::1"):
        _term.field("Note", _term.paint("reachable from other computers: anyone who can reach this "
                                        "port can upload a file", "warn"))
    threading.Thread(target=preload, daemon=True).start()  # warm atlases in the background
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _term.field("Stopped", "")
    finally:
        srv.server_close()
