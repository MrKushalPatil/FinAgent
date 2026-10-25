"""
Generate the richer *synthetic* demo statement (PDF + CSV): samples/demo_statement.pdf / .csv

Designed so ONE statement produces three different affordability verdicts and shows every feature:
salary + bonus, car-loan EMI, SIP, rent, cash withdrawals, UPI ids and a full account number in a narration
(to demonstrate masking), a multi-page PDF, and six months of data (Apr-Sep 2026). No real data is used.
"""
import csv
import random
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)
random.seed(7)
START_BALANCE = 61500.00
MONTHS = [(2026, m) for m in range(4, 10)]


def r(a, b):
    return round(random.uniform(a, b), 2)


def build_rows():
    rows = []
    for (y, m) in MONTHS:
        D = lambda d: date(y, m, d)
        rows += [
            (D(1), "NEFT-SALARY NOVA SOFTWARE SERVICES LTD", 0, 110000.00),
            (D(2), "UPI/Rent to Mr Joshi", 24000.00, 0),
            (D(4), "NACH-CAR LOAN EMI ICICI BANK", 8500.00, 0),
            (D(5), "ACH-SIP HDFC Mutual Fund", 6000.00, 0),
            (D(7), "BillPay MSEDCL Electricity", r(1500, 3200), 0),
            (D(8), "Airtel Broadband Bill", 999.00, 0),
            (D(8), "Jio Prepaid Recharge", 399.00, 0),
            (D(9), "UPI/BigBasket Groceries", r(2400, 3800), 0),
            (D(12), "UPI/Swiggy Food Order", r(350, 850), 0),
            (D(14), "ATM CASH WDL PIMPRI CHINCHWAD", 4000.00, 0),
            (D(15), "UPI/Amazon Pay Shopping", r(900, 4800), 0),
            (D(17), "UPI/Uber India Ride", r(160, 540), 0),
            (D(19), "UPI/Zomato Restaurant", r(400, 1300), 0),
            (D(21), "UPI/DMart Grocery", r(1800, 3000), 0),
            (D(23), "UPI/HPCL Petrol Pump", r(1600, 2600), 0),
            (D(25), "Netflix Subscription", 649.00, 0),
            (D(28), "UPI/Apollo Pharmacy", r(250, 1100), 0),
        ]
        if m in (6, 9):
            rows.append((D(16), "LIC Premium Payment", 12000.00, 0))
            rows.append((D(30), "Interest Credit SB Account", r(900, 1500), 0 if False else 0))
        if m == 5:
            rows.append((D(11), "UPI/priya.s@ybl Birthday Gift", 2500.00, 0))
        if m == 7:
            rows.append((D(10), "NEFT-ANNUAL BONUS NOVA SOFTWARE", 0, 25000.00))
            rows.append((D(13), "UPI/rahul.k@okaxis Dinner Split", 0, 1800.00))
        if m == 8:
            rows.append((D(6), "NEFT-TO A/C 123456789012 URBAN FURNITURE MART", 38000.00, 0))
    # interest rows above were appended as debit by mistake of tuple order: fix to credits
    fixed = []
    for d, n, dr, cr in rows:
        if n.startswith("Interest Credit"):
            dr, cr = 0, round(random.uniform(900, 1500), 2)
        fixed.append((d, n, dr, cr))
    fixed.sort(key=lambda x: x[0])
    return fixed


def with_balance(rows):
    bal, out = START_BALANCE, []
    for d, n, dr, cr in rows:
        bal = round(bal - dr + cr, 2)
        out.append((d, n, dr, cr, bal))
    return out


fmt = lambda x: f"{x:,.2f}"


def write_csv(rows):
    with open(OUT / "demo_statement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Narration", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
        for d, n, dr, cr, bal in rows:
            w.writerow([d.strftime("%d-%m-%Y"), n, fmt(dr) if dr else "", fmt(cr) if cr else "", fmt(bal)])


def write_pdf(rows):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    doc = SimpleDocTemplate(str(OUT / "demo_statement.pdf"), pagesize=A4, leftMargin=24, rightMargin=24,
                            topMargin=30, bottomMargin=30)
    st = getSampleStyleSheet()
    story = [Paragraph("DEMO BANK OF INDIA - Savings Account Statement", st["Title"]),
             Paragraph("<b>SYNTHETIC DATA FOR DEMONSTRATION ONLY - not a real account</b>", st["Normal"]),
             Paragraph("Account holder: Demo User &nbsp;&nbsp; Account No: 123456789012 &nbsp;&nbsp; IFSC: DEMO0001234", st["Normal"]),
             Paragraph("Statement period: 01-04-2026 to 30-09-2026", st["Normal"]),
             Paragraph(f"Opening Balance: {fmt(START_BALANCE)}", st["Normal"]), Spacer(1, 10)]
    data = [["Date", "Narration", "Withdrawal", "Deposit", "Balance"]]
    for d, n, dr, cr, bal in rows:
        data.append([d.strftime("%d-%m-%Y"), n, fmt(dr) if dr else "", fmt(cr) if cr else "", fmt(bal)])
    t = Table(data, colWidths=[62, 250, 70, 70, 80], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")])]))
    story.append(t)
    story += [Spacer(1, 10), Paragraph("This is a computer-generated synthetic statement.", st["Italic"])]
    doc.build(story)


if __name__ == "__main__":
    rows = with_balance(build_rows())
    write_csv(rows)
    write_pdf(rows)
    print(f"{len(rows)} transactions written to {OUT}")
