import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
OPEC_DIR = BASE_DIR / "data" / "reports" / "energy" / "opec"
RAW_FILE = OPEC_DIR / "opec_texts_raw.csv"
OUT_FILE = OPEC_DIR / "opec_texts_clean_sections.csv"


# --- 1. Cleaning ---
def clean_text(text: str) -> str:
    """Remove headers, page numbers, and excessive whitespace."""
    text = re.sub(r"\n\s*\n+", "\n", text)        # collapse blank lines
    text = re.sub(r"Page\s*\d+\s*of\s*\d+", "", text, flags=re.I)
    text = re.sub(r"\n\d+\n", "\n", text)         # stray page numbers
    text = re.sub(r"\s{2,}", " ", text)           # multiple spaces
    return text.strip()


# --- 2. Section splitting ---
SECTION_HEADERS = [
    "World Oil Demand",
    "World Oil Supply",
    "Non-OPEC Supply",
    "OPEC Crude Oil Production",
    "Balance of Supply and Demand",
    "Product Markets and Refining Operations",
    "Crude and Product Prices",
]


def split_into_sections(text: str):
    """Return dict {section_name: section_text}."""
    sections = {}
    pattern = r"(" + "|".join([re.escape(s) for s in SECTION_HEADERS]) + r")"
    parts = re.split(pattern, text)
    current = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in SECTION_HEADERS:
            current = part
            sections[current] = ""
        elif current:
            sections[current] += part + " "
    return sections


# --- 3. Main driver ---
def main():
    df = pd.read_csv(RAW_FILE)
    records = []

    print(f"Cleaning and segmenting {len(df)} reports...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cleaned = clean_text(str(row["text"]))
        sections = split_into_sections(cleaned)
        for sec, content in sections.items():
            records.append(
                {
                    "filename": row["filename"],
                    "report_date": row.get("report_date"),
                    "year": row.get("year"),
                    "month_name": row.get("month_name"),
                    "section": sec,
                    "content": content.strip(),
                }
            )

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nCleaned and segmented texts saved -> {OUT_FILE}")
    print(f"Total segmented rows: {len(out_df)}")


if __name__ == "__main__":
    main()
