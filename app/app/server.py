"""
FinAgent web server (standard library only).

  GET  /                      -> UI
  GET  /api/health            -> model / readiness
  GET  /api/tools             -> tool schemas shown to the LLM (great for the jury: "this is what the model sees")
  POST /api/upload?session=&filename=   (raw file bytes)  -> {file_id}
  POST /api/chat              {session, message}  -> Server-Sent Events stream of agent steps
  POST /api/reset             {session}
  GET  /api/guardrail-demo    -> failure cases with / without guardrails
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import registry
from .agent import FinAgent
from .config import settings
from .demo_cases import run_cases
from .guardrails import ToolInputError, check_user_message, redact
from .llm import make_llm
from .rag import get_retriever
from .registry import ToolContext

STATIC = Path(__file__).parent / "static"
SESSION_RE = re.compile(r"^[A-Za-z0-9\-]{8,64}$")


class Session:
    def __init__(self):
        self.history: list[dict] = []
        self.ctx = ToolContext()
        self.lock = threading.Lock()


SESSIONS: dict[str, Session] = {}
_SESS_LOCK = threading.Lock()


def get_session(sid: str) -> Session:
    with _SESS_LOCK:
        if len(SESSIONS) > 500:  # crude memory cap for a demo server
            SESSIONS.pop(next(iter(SESSIONS)))
        return SESSIONS.setdefault(sid, Session())


class Handler(BaseHTTPRequestHandler):
    server_version = "FinAgent/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # ---- helpers -------------------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode())

    def _body(self, limit: int) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ToolInputError(f"Request too large (limit {limit // 1024} KB).")
        return self.rfile.read(n)

    # ---- routes --------------------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(200, {"ready": settings.llm_ready, "model": "mock (not an LLM)" if settings.provider == "mock" else settings.model,
                             "provider": settings.provider, "web_search": settings.enable_web,
                             "limits": {"max_steps": settings.max_steps, "max_tool_calls": settings.max_total_tool_calls}})
        elif path == "/api/tools":
            self._json(200, registry.tool_specs())
        elif path == "/api/rag/info":
            self._json(200, get_retriever().stats())
        elif path == "/api/rag/chunks":
            self._json(200, [{"id": c.id, "doc": c.doc_title, "sections": c.sections, "overlap": c.overlap_text,
                              "core": c.core_text, "distance_before": c.distance_before, "source": c.source_url}
                             for c in get_retriever().chunks])
        elif path == "/api/guardrail-demo":
            self._json(200, run_cases())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                self._upload(parse_qs(parsed.query))
            elif parsed.path == "/api/chat":
                self._chat()
            elif parsed.path == "/api/rag/search":
                data = json.loads(self._body(4096) or b"{}")
                query = check_user_message(str(data.get("query", ""))[:300], 300)
                self._json(200, get_retriever().search(query, k=3))
            elif parsed.path == "/api/use-sample":
                self._use_sample()
            elif parsed.path == "/api/reset":
                data = json.loads(self._body(4096) or b"{}")
                sid = data.get("session", "")
                if SESSION_RE.match(sid):
                    SESSIONS.pop(sid, None)
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except ToolInputError as e:
            self._json(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # never leak stack traces to the client
            self._json(500, {"error": f"Server error ({type(e).__name__})."})

    # ---- upload --------------------------------------------------------------------------
    def _upload(self, q: dict):
        sid = (q.get("session") or [""])[0]
        if not SESSION_RE.match(sid):
            raise ToolInputError("Invalid session.")
        name = Path((q.get("filename") or ["statement"])[0]).name[:80]
        ext = Path(name).suffix.lower()
        if ext not in (".pdf", ".csv"):
            raise ToolInputError("Only PDF or CSV bank statements are supported.")
        data = self._body(settings.max_upload_bytes)
        if not data:
            raise ToolInputError("The file is empty.")
        if ext == ".pdf" and not data.startswith(b"%PDF"):
            raise ToolInputError("This file is not a valid PDF.")
        if ext == ".csv" and b"\x00" in data[:2048]:
            raise ToolInputError("This file does not look like a CSV.")
        sess = get_session(sid)
        if len(sess.ctx.files) >= 5:
            raise ToolInputError("Upload limit reached for this session (5 files). Start a new chat.")
        file_id = "f_" + uuid.uuid4().hex[:8]
        folder = settings.data_dir / "uploads" / sid
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{file_id}{ext}"
        path.write_bytes(data)
        sess.ctx.files[file_id] = {"path": path, "name": redact(name)}
        self._json(200, {"file_id": file_id, "name": name, "bytes": len(data)})

    def _use_sample(self):
        """Attach the bundled synthetic statement to the session (handy for demos)."""
        data = json.loads(self._body(4096) or b"{}")
        sid = data.get("session", "")
        if not SESSION_RE.match(sid):
            raise ToolInputError("Invalid session.")
        src = settings.data_dir.parent / "samples" / "sample_statement.pdf"
        if not src.exists():
            src = src.with_suffix(".csv")
        if not src.exists():
            raise ToolInputError("Sample statement not found. Run: python scripts/make_sample_statement.py")
        sess = get_session(sid)
        file_id = "f_" + uuid.uuid4().hex[:8]
        folder = settings.data_dir / "uploads" / sid
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{file_id}{src.suffix}"
        path.write_bytes(src.read_bytes())
        sess.ctx.files[file_id] = {"path": path, "name": src.name}
        self._json(200, {"file_id": file_id, "name": src.name, "bytes": path.stat().st_size})

    # ---- chat (SSE) ----------------------------------------------------------------------
    def _sse(self, event: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(event, ensure_ascii=False, default=str).encode() + b"\n\n")
        self.wfile.flush()

    def _chat(self):
        data = json.loads(self._body(32 * 1024) or b"{}")
        sid = data.get("session", "")
        if not SESSION_RE.match(sid):
            raise ToolInputError("Invalid session.")
        message = check_user_message(data.get("message", ""), settings.max_message_chars)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        if not settings.llm_ready:
            self._sse({"type": "error", "message": "No LLM API key configured. Copy .env.example to .env, "
                                                    "add your key, and restart (see README)."})
            return
        sess = get_session(sid)
        with sess.lock:
            agent = FinAgent(make_llm())
            for event in agent.run(message, sess.history, sess.ctx):
                self._sse(event)
                if event["type"] == "final":
                    sess.history += [{"role": "user", "content": message},
                                     {"role": "assistant", "content": event["content"]}]
                    del sess.history[:-20]


def serve() -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", settings.port), Handler)
    srv.daemon_threads = True
    mode = "MOCK (rule-based, not an LLM)" if settings.provider == "mock" else settings.model
    print(f"FinAgent running →  http://localhost:{settings.port}   |   model: {mode}")
    if not settings.llm_ready:
        print("⚠  No API key set. Copy .env.example to .env and add LLM_API_KEY (or set LLM_PROVIDER=mock to test the UI).")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
