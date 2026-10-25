"""
DEV/TEST ONLY — a rule-based stand-in with the same interface as LLMClient.
It is NOT a language model; it uses keyword rules so the UI and tool pipeline can be exercised without an API key.
Never present it as the real agent in a demo.
"""
from __future__ import annotations

import json
import re


def _money(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr)\b", text, re.I)
    if m:
        return float(m.group(1)) * (100000 if m.group(2).lower() in ("lakh", "lac") else 10000000)
    m = re.search(r"\b(\d{5,9})\b", text.replace(",", ""))
    return float(m.group(1)) if m else None


class MockLLM:
    model = "mock-rule-based (not an LLM)"

    @staticmethod
    def _call(i: int, name: str, args: dict) -> dict:
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"mock_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    def chat(self, messages: list[dict], tools=None) -> dict:
        system = messages[0]["content"]
        last_user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user = messages[last_user_idx]["content"].lower()
        results = [(m.get("name"), json.loads(m["content"])) for m in messages[last_user_idx + 1:] if m["role"] == "tool"]
        done = {n for n, _ in results}
        fid = re.search(r"file_id=(f_[0-9a-f]{8})", system)

        wants_stmt = any(w in user for w in ("statement", "spend", "income", "salary", "savings", "afford", "expense"))
        wants_emi = any(w in user for w in ("emi", "loan", "afford", "instal"))
        rate = re.search(r"(\d+(?:\.\d+)?)\s*%", user)
        ten = re.search(r"(\d+)\s*(year|yr|month)", user)
        amount = _money(user)

        if wants_stmt and fid and "analyze_bank_statement" not in done:
            return self._call(1, "analyze_bank_statement", {"file_id": fid.group(1)})
        if wants_emi and amount and rate and ten and "calculate_emi" not in done:
            args = {"principal": amount, "annual_rate_percent": float(rate.group(1)),
                    "tenure_value": int(ten.group(1)), "tenure_unit": "months" if ten.group(2) == "month" else "years"}
            stmt = dict(results).get("analyze_bank_statement")
            if stmt:
                a = stmt["averages_per_month"]
                args["monthly_income"] = a["income"]
                args["existing_monthly_obligations"] = a["existing_emi_or_loan_payments"]
            return self._call(2, "calculate_emi", args)
        if not results and any(w in user for w in ("ppf", "sukanya", "tax", "sip", "nps", "epf", "pmay", "scheme",
                                                    "mutual", "deposit", "fraud", "credit score")):
            return self._call(3, "search_financial_info", {"query": user[:200]})
        if not results and re.search(r"\d\s*[\+\-\*/x]\s*\d", user):
            expr = re.sub(r"[^0-9+\-*/(). ]", "", user.replace("x", "*"))
            return self._call(4, "calculator", {"expression": expr.strip()})

        if results:
            lines = []
            for name, res in results:
                if "error" in res:
                    lines.append(f"- {name}: {res['error']}")
                elif "summary_line" in res:
                    lines.append(f"- **{name}**: {res['summary_line']}")
                    if name == "calculate_emi" and "affordability" in res:
                        lines.append(f"  Affordability: {res['affordability']['verdict']}")
                    if name == "search_financial_info" and res.get("knowledge_base"):
                        lines.append(f"  {res['knowledge_base'][0]['text'][:300]}…")
            return {"role": "assistant", "content": "_(offline test mode — rule-based, not an LLM)_\n" + "\n".join(lines)}
        return {"role": "assistant", "content": "_(offline test mode)_ I can help with EMI, statements and finance "
                                                "questions. Try: “EMI for 50 lakh at 8.5% for 20 years”."}
