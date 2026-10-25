"""Generate a realistic *synthetic* bank statement (PDF + CSV) for demos. No real data is used."""
import csv
import random
import sys
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples"
OUT.mkdir(exist_ok=True)
random.seed(42)

START_BALANCE = 42000.00
MONTHS = [(2025, m) for m in range(4, 10)]  # Apr–Sep 2025


def build_rows():
    rows = []  # (date, narration, debit, credit)
    for (y, m) in MONTHS:
        rows.append((date(y, m, 1), "NEFT-SALARY ACME TECHNOLOGIES PVT LTD", 0, 85000.00))
        rows.append((date(y, m, 3), "UPI/Rent to Landlord Mr Deshmukh", 22000.00, 0))
        rows.append((date(y, m, 5), "NACH-CAR LOAN EMI HDFC BANK", 8500.00, 0))
        rows.append((date(y, m, 6), "ACH-SIP Mutual Fund Groww", 5000.00, 0))
        rows.append((date(y, m, 8), "BillPay MSEDCL Electricity", round(random.uniform(1400, 2600), 2), 0))
        rows.append((date(y, m, 9), "Airtel Broadband Bill", 1179.00, 0))
        rows.append((date(y, m, 10), "UPI/BigBasket Groceries", round(random.uniform(2200, 3800), 2), 0))
        rows.append((date(y, m, 12), "UPI/Swiggy Food Order", round(random.uniform(350, 900), 2), 0))
        rows.append((date(y, m, 14), "ATM CASH WDL PIMPRI", 5000.00, 0))
        rows.append((date(y, m, 16), "UPI/Amazon Pay Shopping", round(random.uniform(800, 4200), 2), 0))
        rows.append((date(y, m, 18), "UPI/Uber India Ride", round(random.uniform(180, 520), 2), 0))
        rows.append((date(y, m, 20), "UPI/DMart Grocery", round(random.uniform(1800, 3200), 2), 0))
        rows.append((date(y, m, 22), "UPI/Zomato Restaurant", round(random.uniform(300, 1200), 2), 0))
        rows.append((date(y, m, 25), "UPI/HPCL Petrol Pump", round(random.uniform(1500, 2500), 2), 0))
        if m in (6, 9):
            rows.append((date(y, m, 15), "LIC Premium Payment", 12000.00, 0))
        if m == 7:
            rows.append((date(y, m, 11), "Amazon Refund Credit", 0, 1999.00))
        if m == 8:
            rows.append((date(y, m, 27), "UPI/Weekend Trip Booking MakeMyTrip", 14500.00, 0))
    rows.sort(key=lambda r: r[0])
    return rows


def with_balance(rows):
    bal, out = START_BALANCE, []
    for d, narr, dr, cr in rows:
        bal = round(bal - dr + cr, 2)
        out.append((d, narr, dr, cr, bal))
    return out


def fmt(x):
    return f"{x:,.2f}"


def write_csv(rows):
    with open(OUT / "sample_statement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Narration", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
        for d, narr, dr, cr, bal in rows:
            w.writerow([d.strftime("%d-%m-%Y"), narr, fmt(dr) if dr else "", fmt(cr) if cr else "", fmt(bal)])


def write_pdf(rows):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    doc = SimpleDocTemplate(str(OUT / "sample_statement.pdf"), pagesize=A4, leftMargin=24, rightMargin=24,
                            topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [Paragraph("DEMO BANK OF INDIA - Account Statement (SYNTHETIC DATA)", styles["Title"]),
             Paragraph("Account holder: Demo User &nbsp;&nbsp; Statement period: 01-04-2025 to 30-09-2025", styles["Normal"]),
             Paragraph(f"Opening Balance: {fmt(START_BALANCE)}", styles["Normal"]), Spacer(1, 10)]
    data = [["Date", "Narration", "Withdrawal", "Deposit", "Balance"]]
    for d, narr, dr, cr, bal in rows:
        data.append([d.strftime("%d-%m-%Y"), narr, fmt(dr) if dr else "", fmt(cr) if cr else "", fmt(bal)])
    t = Table(data, colWidths=[62, 250, 70, 70, 80], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
    ]))
    story.append(t)
    doc.build(story)


if __name__ == "__main__":
    rows = with_balance(build_rows())
    write_csv(rows)
    try:
        write_pdf(rows)
    except ImportError:
        print("reportlab not installed: only the CSV was generated (pip install reportlab for the PDF).", file=sys.stderr)
    print(f"Wrote samples to {OUT}")
