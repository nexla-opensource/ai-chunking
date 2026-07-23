"""Generate small deterministic fixture files for document_chunker tests.

Idempotent: fixed literal data only, no timestamps or randomness.
Run: /Users/tariq/Documents/ai-chunking/.venv/bin/python make_fixtures.py
"""
import os

import pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

CSV_ROWS = [
    ("2024-01-05", "1000-CASH", "Opening balance", "0.00", "12500.00"),
    ("2024-01-12", "4000-REV", "Invoice INV-1001", "2500.00", "15000.00"),
    ("2024-01-19", "6100-RENT", "Office rent January", "-1800.00", "13200.00"),
    ("2024-02-02", "4000-REV", "Invoice INV-1002", "3100.00", "16300.00"),
    ("2024-02-14", "6200-UTIL", "Utilities February", "-240.50", "16059.50"),
    ("2024-02-28", "6300-PAYR", "Payroll February", "-5200.00", "10859.50"),
    ("2024-03-08", "4000-REV", "Invoice INV-1003", "2750.00", "13609.50"),
    ("2024-03-29", "6100-RENT", "Office rent March", "-1800.00", "11809.50"),
]


def make_csv(path):
    lines = ["date,account,description,amount,balance"]
    for r in CSV_ROWS:
        lines.append(",".join(r))
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


def make_xlsx(path):
    summary = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03", "Q1 Total", "Average"],
            "revenue": [2500.00, 3100.00, 2750.00, 8350.00, 2783.33],
            "expenses": [1800.00, 5440.50, 1800.00, 9040.50, 3013.50],
            "net": [700.00, -2340.50, 950.00, -690.50, -230.17],
        }
    )
    detail = pd.DataFrame(
        CSV_ROWS, columns=["date", "account", "description", "amount", "balance"]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        detail.to_excel(writer, sheet_name="Detail", index=False)


def make_png(path):
    img = Image.new("RGB", (400, 160), "white")
    draw = ImageDraw.Draw(img)
    lines = [
        "QUARTERLY REPORT",
        "Q1 2024 (ending 2024-03-31)",
        "Total revenue: $1,250,000",
    ]
    y = 30
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 35
    img.save(path)


def main():
    make_csv(os.path.join(HERE, "tiny.csv"))
    make_xlsx(os.path.join(HERE, "tiny.xlsx"))
    make_png(os.path.join(HERE, "tiny.png"))
    print("fixtures written to", HERE)


if __name__ == "__main__":
    main()
