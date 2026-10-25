"""Tool: bank statement analyzer (PDF or CSV) — parses, categorises and summarises cash-flow."""
from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..guardrails import ToolInputError, redact
from ..util import inr

NAME = "analyze_bank_statement"

DESCRIPTION = (
    "Parses an uploaded bank statement (PDF or CSV) and returns verified cash-flow facts: average monthly income, "
    "expenses, net surplus, savings rate, existing EMI/loan burden, spending categories and largest expenses. "
    "Use it when the user asks about their own income, spending, savings, or loan eligibility based on their "
    "statement. It needs a file_id from the 'Uploaded files' list in the system message; if no file is uploaded, "
    "ask the user to upload one instead of calling this tool. Typically call this BEFORE calculate_emi when "
    "judging loan affordability."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "file_id": {
            "type": "string", "maxLength": 20,
            "description": "The file_id of an uploaded statement exactly as listed in the system message, e.g. 'f_1a2b3c4d'.",
        },
    },
    "required": ["file_id"],
}
MAX_CALLS = 2

# ---------------------------------------------------------------------------------------------
# Categorisation rules (first match wins; order matters)
# ---------------------------------------------------------------------------------------------
CATEGORY_RULES = [
    ("Salary", r"salary|sal cr|payroll|wages|stipend"),
    ("EMI / Loan", r"\bemi\b|loan|nach|ecs|equated|repayment|bajaj fin|hdfc ltd|lic hfl"),
    ("Rent", r"\brent\b|landlord|housing society|maintenance"),
    ("Investments", r"\bsip\b|mutual fund|zerodha|groww|upstox|\bppf\b|\bnps\b|coin|indmoney|kuvera|smallcase"),
    ("Insurance", r"insurance|premium|\blic\b|policybazaar|star health|icici pru|hdfc life"),
    ("Utilities & Bills", r"electric|mseb|mahadiscom|water bill|gas|broadband|airtel|jio|vodafone|vi prepaid|recharge|bsnl|dth|tata play|bill pay"),
    ("Groceries", r"bigbasket|dmart|d-mart|grocery|blinkit|zepto|instamart|reliance fresh|vegetable|kirana"),
    ("Food & Dining", r"swiggy|zomato|restaurant|cafe|dominos|pizza|mcdonald|starbucks|hotel|food"),
    ("Shopping", r"amazon|flipkart|myntra|ajio|meesho|nykaa|croma|shopping|mall"),
    ("Transport & Fuel", r"uber|\bola\b|rapido|petrol|fuel|hpcl|bpcl|iocl|irctc|metro|fastag|redbus"),
    ("Entertainment", r"netflix|spotify|hotstar|prime video|bookmyshow|youtube|gaming"),
    ("Health", r"pharmacy|apollo|hospital|clinic|medplus|1mg|diagnostic"),
    ("Cash Withdrawal", r"\batm\b|cash wdl|cash withdrawal|nfs"),
    ("Interest / Refund", r"interest|refund|cashback|reversal"),
]
_COMPILED = [(c, re.compile(p, re.I)) for c, p in CATEGORY_RULES]
CREDIT_HINT = re.compile(r"salary|credit|refund|interest|cashback|reversal|\bby\b|deposit|neft cr|imps cr", re.I)


def categorize(desc: str, is_credit: bool) -> str:
    for cat, rx in _COMPILED:
        if rx.search(desc):
            if is_credit and cat not in ("Salary", "Interest / Refund"):
                continue
            return cat
    return "Other Income" if is_credit else "UPI / Transfers & Other"


# ---------------------------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------------------------
DATE_FORMATS = ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
                "%d %b %Y", "%d-%b-%Y", "%d %b %y", "%d-%b-%y", "%Y-%m-%d"]
_DATE = r"(?P<date>\d{4}-\d{2}-\d{2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[ -][A-Za-z]{3}[ -]\d{2,4})"
_AMT = r"-?(?:₹|INR|Rs\.?)?\s?[\d,]+\.\d{2}"
LINE_RE = re.compile(
    rf"^\s*{_DATE}\s+(?P<desc>.+?)\s+(?P<amt>{_AMT})\s*(?P<amt_t>Dr|Cr)?\s+(?P<bal>{_AMT})\s*(?P<bal_t>Dr|Cr)?\s*$",
    re.I)
OPENING_RE = re.compile(r"opening\s+balance\D{0,15}([\d,]+\.\d{2})", re.I)


def _num(s: str) -> float:
    return float(re.sub(r"[^\d.\-]", "", s))


def _parse_date(s: str):
    s = re.sub(r"\s+", " ", s.replace(",", " ")).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_pdf_text(text: str) -> list[dict]:
    opening = None
    m = OPENING_RE.search(text)
    if m:
        opening = _num(m.group(1))
    txns, prev_bal = [], opening
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        d = _parse_date(m.group("date"))
        if not d:
            continue
        amt, bal = abs(_num(m.group("amt"))), _num(m.group("bal"))
        desc = m.group("desc").strip()
        is_credit = None
        if prev_bal is not None:
            if abs(prev_bal - amt - bal) < 0.02:
                is_credit = False
            elif abs(prev_bal + amt - bal) < 0.02:
                is_credit = True
        if is_credit is None and m.group("amt_t"):
            is_credit = m.group("amt_t").lower() == "cr"
        if is_credit is None:
            is_credit = bool(CREDIT_HINT.search(desc))
        txns.append({"date": d, "desc": desc, "amount": amt, "credit": is_credit, "balance": bal})
        prev_bal = bal
    return txns


def _read_pdf(path: Path) -> list[dict]:
    try:
        import pdfplumber
    except ImportError:
        raise ToolInputError("PDF support is not installed on the server (pip install pdfplumber).")
    try:
        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages[:60])
    except Exception as e:  # encrypted, corrupt, ...
        raise ToolInputError(f"The PDF could not be opened ({type(e).__name__}). It may be password-protected or damaged.")
    if not text.strip():
        raise ToolInputError("The PDF has no extractable text (it may be a scanned image). Ask the user for a text-based PDF or CSV.")
    txns = _parse_pdf_text(text)
    if not txns:
        raise ToolInputError("No transactions were recognised. Supported layout: Date, Description, Amount (debit/credit), Balance on one line.")
    return txns


def _find(headers: list[str], *candidates: str):
    for h in headers:
        low = h.lower()
        if any(c in low for c in candidates):
            return h
    return None


def _read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    headers = reader.fieldnames or []
    c_date = _find(headers, "date")
    c_desc = _find(headers, "narration", "description", "particular", "remark", "details")
    c_dr = _find(headers, "debit", "withdrawal", "dr amount")
    c_cr = _find(headers, "credit", "deposit", "cr amount")
    c_amt = _find(headers, "amount")
    c_type = _find(headers, "type", "dr/cr")
    c_bal = _find(headers, "balance")
    if not (c_date and c_desc and (c_dr or c_cr or c_amt)):
        raise ToolInputError(f"CSV needs Date, Description and Debit/Credit (or Amount) columns. Found: {headers}")
    txns = []
    for row in reader:
        d = _parse_date((row.get(c_date) or "").strip())
        if not d:
            continue
        desc = (row.get(c_desc) or "").strip()
        dr = _num(row[c_dr]) if c_dr and (row.get(c_dr) or "").strip() else 0.0
        cr = _num(row[c_cr]) if c_cr and (row.get(c_cr) or "").strip() else 0.0
        if dr or cr:
            amount, credit = (cr, True) if cr and not dr else (dr, False)
        elif c_amt and (row.get(c_amt) or "").strip():
            a = _num(row[c_amt])
            t = (row.get(c_type) or "").lower() if c_type else ""
            credit = a > 0 if not t else t.startswith("c")
            amount = abs(a)
        else:
            continue
        bal = _num(row[c_bal]) if c_bal and (row.get(c_bal) or "").strip() else None
        txns.append({"date": d, "desc": desc, "amount": abs(amount), "credit": credit, "balance": bal})
    if not txns:
        raise ToolInputError("No transactions could be read from the CSV.")
    return txns


# ---------------------------------------------------------------------------------------------
# Tool entry points
# ---------------------------------------------------------------------------------------------
def validate(args: dict, ctx) -> dict:
    fid = args["file_id"]
    if not re.fullmatch(r"f_[0-9a-f]{8}", fid):
        raise ToolInputError(f"'{fid}' is not a valid file_id. Use an id exactly as listed under 'Uploaded files'.")
    if fid not in ctx.files:
        known = list(ctx.files) or "none"
        raise ToolInputError(f"Unknown file_id '{fid}'. Uploaded files: {known}. "
                             "If nothing is uploaded, ask the user to upload their statement.")
    return args


def run(args: dict, ctx) -> dict:
    meta = ctx.files[args["file_id"]]
    path: Path = meta["path"]
    txns = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_csv(path)
    txns.sort(key=lambda t: t["date"])
    for t in txns:
        t["category"] = categorize(t["desc"], t["credit"])
        t["desc"] = redact(t["desc"])

    months = defaultdict(lambda: {"income": 0.0, "expenses": 0.0})
    cat_spend: dict[str, float] = defaultdict(float)
    salary_total = credits = debits = emi_total = 0.0
    for t in txns:
        key = t["date"].strftime("%Y-%m")
        if t["credit"]:
            months[key]["income"] += t["amount"]
            credits += t["amount"]
            if t["category"] == "Salary":
                salary_total += t["amount"]
        else:
            months[key]["expenses"] += t["amount"]
            debits += t["amount"]
            cat_spend[t["category"]] += t["amount"]
            if t["category"] == "EMI / Loan":
                emi_total += t["amount"]

    n_months = max(len(months), 1)
    basis = "salary credits" if salary_total > 0 else "all credits"
    income_total = salary_total if salary_total > 0 else credits
    avg_income = income_total / n_months
    avg_exp = debits / n_months
    avg_net = avg_income - avg_exp
    savings_rate = (avg_net / avg_income * 100) if avg_income else 0.0
    avg_emi = emi_total / n_months

    balances = [t["balance"] for t in txns if t["balance"] is not None]
    spend_cats = sorted(cat_spend.items(), key=lambda kv: -kv[1])
    biggest = sorted((t for t in txns if not t["credit"]), key=lambda t: -t["amount"])[:5]

    insights = []
    if savings_rate < 10:
        insights.append(f"Savings rate is low ({savings_rate:.0f}%); aim for 20% or more if possible.")
    elif savings_rate >= 30:
        insights.append(f"Strong savings rate ({savings_rate:.0f}% of income).")
    if avg_income and avg_emi / avg_income > 0.4:
        insights.append("Existing EMIs already exceed 40% of income — new borrowing capacity is limited.")
    cash = cat_spend.get("Cash Withdrawal", 0)
    if debits and cash / debits > 0.15:
        insights.append(f"Cash withdrawals are {cash / debits * 100:.0f}% of spending, which is hard to track.")
    if balances and min(balances) < avg_exp * 0.25:
        insights.append("Balance dipped very low at some point relative to monthly expenses.")
    if len(months) < 3:
        insights.append(f"Only {len(months)} month(s) of data — averages may not be representative.")

    return {
        "file": meta["name"],
        "period": {"from": txns[0]["date"].isoformat(), "to": txns[-1]["date"].isoformat(), "months_covered": len(months)},
        "transactions_parsed": len(txns),
        "income_basis": basis,
        "averages_per_month": {
            "income": round(avg_income), "expenses": round(avg_exp), "net_surplus": round(avg_net),
            "savings_rate_percent": round(savings_rate, 1), "existing_emi_or_loan_payments": round(avg_emi),
            "invested_via_sip_etc": round(cat_spend.get("Investments", 0) / n_months),
            "savings_rate_incl_investments_percent": round(
                (avg_net + cat_spend.get("Investments", 0) / n_months) / avg_income * 100, 1) if avg_income else 0.0,
        },
        "monthly": [{"month": k, "income": round(v["income"]), "expenses": round(v["expenses"]),
                     "net": round(v["income"] - v["expenses"])} for k, v in sorted(months.items())],
        "spending_by_category": [{"category": c, "total": round(a), "share_percent": round(a / debits * 100, 1)}
                                 for c, a in spend_cats[:8]] if debits else [],
        "largest_expenses": [{"date": t["date"].isoformat(), "description": t["desc"][:60],
                              "amount": round(t["amount"]), "category": t["category"]} for t in biggest],
        "balance": {"closing": round(balances[-1]) if balances else None, "lowest": round(min(balances)) if balances else None},
        "insights": insights,
        "display": {"avg_income": inr(avg_income), "avg_expenses": inr(avg_exp), "avg_net_surplus": inr(avg_net),
                    "avg_emi": inr(avg_emi)},
        "summary_line": (f"{len(months)} months · income {inr(avg_income)}/mo · spend {inr(avg_exp)}/mo · "
                         f"surplus {inr(avg_net)}/mo · existing EMIs {inr(avg_emi)}/mo"),
    }
