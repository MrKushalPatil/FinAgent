"""
Guardrails for FinAgent.

Every control here maps to the "Guardrails & Responsible AI" slide:

  1. Tool whitelisting        -> registry.execute() refuses any tool name not registered
  2. Input validation         -> validate_schema() checks types / ranges / enums / unknown args
  3. Execution loop limits    -> LoopGuard (max steps, per-tool budget, duplicate-call blocking)
  4. Data protection          -> redact() masks account numbers, PAN, e-mail, UPI ids
  5. Untrusted tool output    -> sanitize_external_text() strips prompt-injection phrases
  6. Domain preference        -> is_trusted_domain() used by the web-search tool
  7. Financial disclaimer     -> DISCLAIMER appended by the agent
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional


class ToolInputError(Exception):
    """Raised when a tool call is rejected. The message is shown to the LLM so it can self-correct."""


# --------------------------------------------------------------------------------------
# 2. Input validation (a small JSON-Schema subset: type, enum, min/max, maxLength, required)
# --------------------------------------------------------------------------------------
def _coerce(key: str, value: Any, spec: dict) -> Any:
    t = spec.get("type")
    hint = spec.get("description", "")
    if t in ("number", "integer"):
        if isinstance(value, bool):
            raise ToolInputError(f"'{key}' must be a number, got a boolean. {hint}")
        if isinstance(value, str):
            try:
                value = float(value.replace(",", "").strip())
            except ValueError:
                raise ToolInputError(f"'{key}' must be a number, got text {value!r}. {hint}")
        if not isinstance(value, (int, float)):
            raise ToolInputError(f"'{key}' must be a number. {hint}")
        if t == "integer":
            if float(value) != int(value):
                raise ToolInputError(f"'{key}' must be a whole number, got {value}. {hint}")
            value = int(value)
        return value
    if t == "string":
        if not isinstance(value, str):
            raise ToolInputError(f"'{key}' must be a string. {hint}")
        return value.strip()
    if t == "boolean":
        if isinstance(value, bool):
            return value
        raise ToolInputError(f"'{key}' must be true or false. {hint}")
    return value


def validate_schema(schema: dict, args: Any) -> dict:
    """Validate and normalise tool arguments against the tool's JSON schema."""
    if not isinstance(args, dict):
        raise ToolInputError("Arguments must be a JSON object.")
    props = schema.get("properties", {})

    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolInputError(f"Unknown argument(s) {unknown}. Allowed arguments: {sorted(props)}.")

    for req in schema.get("required", []):
        if req not in args or args[req] in (None, ""):
            desc = props.get(req, {}).get("description", "")
            raise ToolInputError(f"Missing required argument '{req}'. {desc}")

    clean: dict = {}
    for key, value in args.items():
        if value is None:
            continue
        spec = props[key]
        value = _coerce(key, value, spec)
        hint = spec.get("description", "")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolInputError(f"'{key}'={value!r} is not allowed. Choose one of {spec['enum']}.")
        if "minimum" in spec and value < spec["minimum"]:
            raise ToolInputError(f"'{key}'={value} is below the minimum {spec['minimum']}. {hint}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ToolInputError(f"'{key}'={value} is above the maximum {spec['maximum']}. {hint}")
        if "maxLength" in spec and isinstance(value, str) and len(value) > spec["maxLength"]:
            raise ToolInputError(f"'{key}' is too long (max {spec['maxLength']} characters).")
        clean[key] = value
    return clean


# --------------------------------------------------------------------------------------
# 3. Execution loop limits
# --------------------------------------------------------------------------------------
def _canonical(name: str, args: Any) -> str:
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        blob = str(args)
    return hashlib.sha1(f"{name}:{blob}".encode()).hexdigest()


class LoopGuard:
    """Stops runaway agents: total-call cap, per-tool cap, and identical-call blocking."""

    def __init__(self, max_total_calls: int = 8):
        self.max_total_calls = max_total_calls
        self.total = 0
        self.per_tool: dict[str, int] = {}
        self.seen: set[str] = set()

    def check(self, name: str, args: Any, per_tool_limit: int) -> Optional[str]:
        if self.total >= self.max_total_calls:
            return (f"Tool-call budget exhausted ({self.max_total_calls} calls). "
                    "Stop calling tools and answer with the information you already have.")
        if self.per_tool.get(name, 0) >= per_tool_limit:
            return (f"'{name}' was already called {per_tool_limit} times in this request. "
                    "Use the earlier results instead of calling it again.")
        if _canonical(name, args) in self.seen:
            return (f"You already called '{name}' with exactly these arguments. "
                    "Reuse the earlier result rather than repeating the call.")
        return None

    def record(self, name: str, args: Any) -> None:
        self.total += 1
        self.per_tool[name] = self.per_tool.get(name, 0) + 1
        self.seen.add(_canonical(name, args))


# --------------------------------------------------------------------------------------
# 4. Data protection
# --------------------------------------------------------------------------------------
_PATTERNS = [
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), "[ID-NUMBER]"),
    (re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+"), "[EMAIL]"),
    (re.compile(r"\b[\w.\-]{2,}@(?!\w+\.)[a-zA-Z]{2,}\b"), "[UPI-ID]"),
    (re.compile(r"\b\d{9,18}\b"), lambda m: "X" * (len(m.group()) - 4) + m.group()[-4:]),
]


def redact(text: str) -> str:
    """Mask personally identifying strings (account numbers, PAN, e-mails, UPI IDs)."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


# --------------------------------------------------------------------------------------
# 5. Untrusted external content (web pages, search snippets, PDF text)
# --------------------------------------------------------------------------------------
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+"
    r"(instructions?|prompts?|rules?)|you\s+are\s+now\s+|system\s+prompt|"
    r"(call|use|invoke)\s+(the\s+)?(tool|function)\s+\w+|reveal\s+your\s+(prompt|instructions)",
    re.IGNORECASE,
)


def sanitize_external_text(text: str, limit: int = 1200) -> str:
    text = _INJECTION.sub("[removed: possible prompt-injection text]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


# --------------------------------------------------------------------------------------
# 6. Domain preference for web results
# --------------------------------------------------------------------------------------
TRUSTED_DOMAINS = (
    "rbi.org.in", "sebi.gov.in", "incometax.gov.in", "incometaxindia.gov.in", "indiapost.gov.in",
    "nsiindia.gov.in", "epfindia.gov.in", "npstrust.org.in", "pfrda.org.in", "finmin.gov.in",
    "financialservices.gov.in", "india.gov.in", "pmaymis.gov.in", "mohua.gov.in", "dicgc.org.in",
    "amfiindia.com", "nseindia.com", "bseindia.com", "cibil.com", "uidai.gov.in", "npci.org.in",
)


def is_trusted_domain(url: str) -> bool:
    m = re.match(r"https?://([^/:]+)", url.strip().lower())
    if not m:
        return False
    host = m.group(1)
    if host.endswith(".gov.in") or host.endswith(".nic.in"):
        return True
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


# --------------------------------------------------------------------------------------
# 7. Disclaimer & input checks
# --------------------------------------------------------------------------------------
DISCLAIMER = ("_Informational assistance only — not professional financial, tax or legal advice. "
              "Verify figures with your bank or a SEBI-registered adviser before acting._")


def check_user_message(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        raise ToolInputError("Please type a question.")
    if len(text) > max_chars:
        raise ToolInputError(f"Message too long ({len(text)} characters; the limit is {max_chars}).")
    return text
