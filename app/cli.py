"""Terminal version of FinAgent — shows every tool call. Usage:  python cli.py [--file samples/sample_statement.pdf]"""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from app.agent import FinAgent
from app.config import settings
from app.llm import make_llm
from app.registry import ToolContext

ap = argparse.ArgumentParser()
ap.add_argument("--file", help="bank statement (PDF/CSV) to make available to the agent")
args = ap.parse_args()

if not settings.llm_ready:
    sys.exit("No API key: copy .env.example to .env and set LLM_API_KEY (or LLM_PROVIDER=mock).")

ctx, history = ToolContext(), []
if args.file:
    src = Path(args.file)
    fid = "f_" + uuid.uuid4().hex[:8]
    dest = settings.data_dir / "uploads" / "cli" / f"{fid}{src.suffix.lower()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    ctx.files[fid] = {"path": dest, "name": src.name}
    print(f"[loaded {src.name} as {fid}]")

agent = FinAgent(make_llm())
print("FinAgent CLI — type 'exit' to quit.\n")
while True:
    try:
        q = input("you › ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("exit", "quit"):
        break
    if not q:
        continue
    for ev in agent.run(q, history, ctx):
        if ev["type"] == "tool_call":
            print(f"  ▸ step {ev['step']}: model calls {ev['name']}({ev['arguments']})")
        elif ev["type"] == "tool_result":
            status = "ok" if ev["ok"] else f"BLOCKED by {ev['guardrail']}: {ev['error']}"
            line = (ev["result"] or {}).get("summary_line", "") if ev["ok"] else ""
            print(f"    ◂ {status} {line} ({ev['ms']} ms)")
        elif ev["type"] == "guardrail":
            print(f"  ⚠ {ev['message']}")
        elif ev["type"] == "error":
            print(f"  ✗ {ev['message']}")
        elif ev["type"] == "final":
            print(f"\nfinagent › {ev['content']}\n")
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": ev["content"]}]
