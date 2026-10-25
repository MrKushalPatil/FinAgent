"""
Tool registry + guarded dispatcher.

`execute()` is the single choke-point through which every LLM-requested action passes:

   whitelist -> parse JSON -> schema validation -> loop guard -> semantic validation -> run -> sanitise output

The LLM never executes anything itself; it only *requests* a call. This code decides whether to run it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from .guardrails import LoopGuard, ToolInputError, validate_schema
from .tools import analyzer, calculator, emi, search


@dataclass
class ToolContext:
    """Per-session state visible to tools. `files` maps server-generated file_id -> {path, name}."""
    files: dict[str, dict] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    module: ModuleType
    max_calls: int

    def spec(self) -> dict:
        """OpenAI-style function-calling spec — this is what the LLM 'sees' and decides from."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _tool(m: ModuleType) -> Tool:
    return Tool(m.NAME, m.DESCRIPTION, m.SCHEMA, m, m.MAX_CALLS)


TOOLS: dict[str, Tool] = {t.name: t for t in (_tool(analyzer), _tool(emi), _tool(search), _tool(calculator))}


def tool_specs() -> list[dict]:
    return [t.spec() for t in TOOLS.values()]


def execute(name: str, raw_args: Any, ctx: ToolContext, loop_guard: Optional[LoopGuard] = None,
            guardrails: bool = True) -> dict:
    """Run one tool call. Always returns a dict: {ok, name, result|error, guardrail, ms}."""
    t0 = time.time()

    def done(ok: bool, **kw) -> dict:
        return {"ok": ok, "name": name, "ms": int((time.time() - t0) * 1000), **kw}

    # 1) Whitelist
    tool = TOOLS.get(name)
    if tool is None:
        return done(False, guardrail="tool_whitelist",
                    error=f"Tool '{name}' does not exist or is not permitted. Allowed tools: {sorted(TOOLS)}.")

    # 2) Parse arguments
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
    except json.JSONDecodeError:
        return done(False, guardrail="input_validation", error="Arguments were not valid JSON. Send a JSON object.")

    if not guardrails:  # used only by the failure demo, to show what happens WITHOUT the controls
        try:
            return done(True, result=tool.module.run(args, ctx), guardrail=None)
        except Exception as e:
            return done(False, guardrail=None, error=f"{type(e).__name__}: {e}")

    try:
        # 3) Schema validation (types, ranges, enums, unknown args)
        args = validate_schema(tool.parameters, args)
        # 4) Loop guard (budget + duplicate-call blocking)
        if loop_guard is not None:
            reason = loop_guard.check(name, args, tool.max_calls)
            if reason:
                return done(False, guardrail="loop_limit", error=reason)
        # 5) Tool-specific semantic validation (cross-field rules, file ownership...)
        args = tool.module.validate(args, ctx)
    except ToolInputError as e:
        return done(False, guardrail="input_validation", error=str(e))

    if loop_guard is not None:
        loop_guard.record(name, args)
    try:
        return done(True, result=tool.module.run(args, ctx), guardrail=None, args=args)
    except ToolInputError as e:   # e.g. unreadable PDF: a user-fixable problem, reported to the LLM
        return done(False, guardrail="input_validation", error=str(e))
    except Exception as e:        # genuine bug / outage: never leak a stack trace to the model
        return done(False, guardrail="tool_failure", error=f"The tool failed unexpectedly ({type(e).__name__}).")
