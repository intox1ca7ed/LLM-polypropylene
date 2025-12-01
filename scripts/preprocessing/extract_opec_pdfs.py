import os
import re
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
OPEC_DIR = BASE_DIR / "data" / "reports" / "energy" / "opec"
OUT_CSV = OPEC_DIR / "opec_texts_raw.csv"

# month helpers
MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_PATTERN = re.compile(
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
    flags=re.I,
)
YEAR_PATTERN = re.compile(r"(20\d{2})")


def parse_month_year_from_filename(fname: str):
    """Try to get month/year from filename like 'MOMR_October_2025.pdf'."""
    name = os.path.splitext(os.path.basename(fname))[0]
    m = MONTH_PATTERN.search(name)
    y = YEAR_PATTERN.search(name)
    if m and y:
        month_txt = m.group(1).lower()
        month_num = MONTH_MAP.get(month_txt)
        year = int(y.group(1))
        return month_num, year
    return None, None


def parse_month_year_from_text(text: str):
    """Fallback: detect month/year from the first 800 chars of text (title pages)."""
    snippet = text[:800]
    m = MONTH_PATTERN.search(snippet)
    y = YEAR_PATTERN.search(snippet)
    if m and y:
        month_txt = m.group(1).lower()
        month_num = MONTH_MAP.get(month_txt)
        year = int(y.group(1))
        return month_num, year
    return None, None


def extract_pdf_text(pdf_path: str) -> tuple[str, int]:
    """Return (full_text, page_count)."""
    text_chunks = []
    pages = 0
    with fitz.open(pdf_path) as doc:
        pages = len(doc)
        for page in doc:
            text_chunks.append(page.get_text("text"))
    return "\n".join(text_chunks), pages


def main():
    records = []
    pdfs = [f for f in os.listdir(OPEC_DIR) if f.lower().endswith(".pdf")]
    pdfs.sort()

    if not pdfs:
        print(f"No PDFs found in: {OPEC_DIR}")
        return

    print(f"Found {len(pdfs)} PDFs. Extracting...")
    for pdf in tqdm(pdfs):
        fpath = OPEC_DIR / pdf

        try:
            full_text, pages = extract_pdf_text(fpath)
        except Exception as e:
            print(f"  Error reading {pdf}: {e}")
            continue

        month_num, year = parse_month_year_from_filename(pdf)
        if not (month_num and year):
            m2, y2 = parse_month_year_from_text(full_text)
            month_num, year = m2, y2

        report_date = None
        month_name = None
        if month_num and year:
            report_date = datetime(year, month_num, 1).strftime("%Y-%m-%d")
            month_name = datetime(year, month_num, 1).strftime("%B")

        records.append(
            {
                "filename": pdf,
                "report_date": report_date,
                "year": year,
                "month_num": month_num,
                "month_name": month_name,
                "pages": pages,
                "text": full_text,
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values(["year", "month_num"], inplace=True, na_position="last")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\nSaved extracted texts -> {OUT_CSV}")
    missing_dates = df["report_date"].isna().sum()
    if missing_dates:
        print(f"  {missing_dates} file(s) missing month/year. Fix filenames or handle in cleaning.")
    else:
        print("All files have parsed month/year.")


if __name__ == "__main__":
    main()
