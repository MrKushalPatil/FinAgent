"""Tool: EMI calculator — a deterministic loan engine (no LLM arithmetic)."""
from __future__ import annotations

from ..guardrails import ToolInputError
from ..util import inr

NAME = "calculate_emi"

DESCRIPTION = (
    "Deterministic loan EMI calculator (reducing-balance method). Use it whenever the user asks about a "
    "loan instalment, total interest, or whether they can afford a loan (home, car, personal, education). "
    "Never compute EMI yourself — always call this tool. If the user's income and existing EMIs are known "
    "(e.g. from analyze_bank_statement), pass them to get the EMI-to-income (FOIR) check. "
    "Do NOT call it if principal, rate or tenure is missing — ask the user instead of guessing."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "principal": {
            "type": "number", "minimum": 10000, "maximum": 100000000,
            "description": "Loan amount in rupees as a full number, e.g. 5000000 for ₹50 lakh (NOT 50 or 5).",
        },
        "annual_rate_percent": {
            "type": "number", "minimum": 1, "maximum": 40,
            "description": "Annual interest rate as a percent number, e.g. 8.5 for 8.5% (NOT 0.085).",
        },
        "tenure_value": {
            "type": "number", "minimum": 1, "maximum": 480,
            "description": "Loan tenure number, interpreted using tenure_unit.",
        },
        "tenure_unit": {
            "type": "string", "enum": ["years", "months"],
            "description": "Unit for tenure_value: 'years' or 'months'.",
        },
        "monthly_income": {
            "type": "number", "minimum": 1000,
            "description": "Optional. Net monthly income in rupees, to compute the EMI-to-income ratio.",
        },
        "existing_monthly_obligations": {
            "type": "number", "minimum": 0,
            "description": "Optional. Existing monthly EMIs/loan payments in rupees (0 if none).",
        },
    },
    "required": ["principal", "annual_rate_percent", "tenure_value", "tenure_unit"],
}
MAX_CALLS = 4


def _months(args: dict) -> int:
    n = args["tenure_value"] * (12 if args["tenure_unit"] == "years" else 1)
    return int(round(n))


def validate(args: dict, ctx) -> dict:
    n = _months(args)
    if n < 3 or n > 480:
        raise ToolInputError(
            f"Tenure of {args['tenure_value']} {args['tenure_unit']} = {n} months is outside 3–480 months "
            "(40 years). Check the unit.")
    return args


def run(args: dict, ctx=None) -> dict:
    P = float(args["principal"])
    annual = float(args["annual_rate_percent"])
    n = _months(args)
    r = annual / 12 / 100

    emi = P / n if r == 0 else P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_payment = emi * n
    total_interest = total_payment - P

    # Year-by-year amortisation summary
    balance, yearly, y_int, y_prin = P, [], 0.0, 0.0
    for m in range(1, n + 1):
        interest = balance * r
        principal_part = emi - interest
        balance = max(balance - principal_part, 0.0)
        y_int += interest
        y_prin += principal_part
        if m % 12 == 0 or m == n:
            yearly.append({"year": (m + 11) // 12, "principal_paid": round(y_prin), "interest_paid": round(y_int),
                           "closing_balance": round(balance)})
            y_int = y_prin = 0.0

    out = {
        "principal": round(P),
        "annual_rate_percent": annual,
        "tenure_months": n,
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest),
        "total_payment": round(total_payment),
        "interest_share_of_total_percent": round(total_interest / total_payment * 100, 1),
        "yearly_summary_first_5_years": yearly[:5],
        "display": {
            "monthly_emi": inr(emi), "total_interest": inr(total_interest),
            "total_payment": inr(total_payment), "principal": inr(P),
        },
    }

    income = args.get("monthly_income")
    if income:
        existing = float(args.get("existing_monthly_obligations") or 0)
        foir = (emi + existing) / income * 100
        verdict = ("comfortable (under 40%)" if foir < 40 else
                   "stretched (40–50%: many lenders may still approve, but budget gets tight)" if foir <= 50 else
                   "high (over 50%: most lenders are likely to decline or reduce the amount)")
        out["affordability"] = {
            "emi_plus_existing_obligations": round(emi + existing),
            "obligation_to_income_percent": round(foir, 1),
            "verdict": verdict,
            "note": "40–50% of net income is a commonly used lender guideline; each lender sets its own limit.",
        }
    out["summary_line"] = f"EMI {inr(emi)}/month · total interest {inr(total_interest)} · {n} months"
    return out
