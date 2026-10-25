"""Step 5 demo: show tool-misuse failures and how the guardrails constrain them. Runs offline, no API key needed."""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.demo_cases import run_cases  # noqa: E402

W = 100
for i, c in enumerate(run_cases(), 1):
    print("=" * W)
    print(f"FAILURE {i}: {c['title']}")
    print(f"  Model requested : {c['model_did']}")
    tag = " (illustrative)" if c.get("illustrative") else ""
    print(f"  WITHOUT guardrails{tag}:")
    print(textwrap.indent(textwrap.fill(c["without"], W - 6), "      "))
    print("  WITH guardrails:")
    print(textwrap.indent(textwrap.fill(c["with"], W - 6), "      "))
    print(f"  Control          : {c['control']}")
print("=" * W)
