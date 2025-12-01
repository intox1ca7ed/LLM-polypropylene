"""Rename OPEC PDF files to a consistent MOMR_<Month>_<Year>.pdf format."""

import re
from pathlib import Path

OPEC_DIR = Path(__file__).resolve().parents[2] / "data" / "reports" / "energy" / "opec"

MONTHS = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}


def main():
    for f in OPEC_DIR.iterdir():
        if f.suffix.lower() != ".pdf":
            continue

        base = f.stem
        m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*", base, re.I)
        y = re.search(r"20\\d{2}", base)
        if not (m and y):
            print("Skipped (month/year not found):", f.name)
            continue

        month = MONTHS[m.group(1).lower()]
        year = y.group(0)
        new_name = f"MOMR_{month}_{year}.pdf"
        new_path = OPEC_DIR / new_name

        if not new_path.exists():
            f.rename(new_path)
            print("Renamed", f.name, "->", new_name)


if __name__ == "__main__":
    main()
