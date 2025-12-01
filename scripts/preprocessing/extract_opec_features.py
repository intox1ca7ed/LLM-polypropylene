import re
from pathlib import Path

import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
OPEC_DIR = BASE_DIR / "data" / "reports" / "energy" / "opec"
CLEAN_FILE = OPEC_DIR / "opec_texts_clean_sections.csv"
OUT_FILE = OPEC_DIR / "opec_features.csv"


# --- 1. Keyword groups ---
KEYWORDS = {
    "supply_up": ["increase", "rise", "growth", "expand", "surge"],
    "supply_down": ["decrease", "cut", "decline", "drop", "reduce", "slowdown"],
    "demand_up": ["demand growth", "strong demand", "recover", "rise"],
    "demand_down": ["weak demand", "sluggish", "decline", "fall", "drop"],
    "price_up": ["higher price", "price increase", "upward trend", "strengthened"],
    "price_down": ["lower price", "price fall", "downward", "weakened"],
}


def keyword_count(text, word_list):
    """Count total matches for a given keyword list."""
    count = 0
    for w in word_list:
        count += len(re.findall(r"\b" + re.escape(w) + r"\b", text, flags=re.I))
    return count


# --- 2. Sentiment & keyword extraction ---
def extract_features(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    kw_counts = {k: keyword_count(text, v) for k, v in KEYWORDS.items()}
    return sentiment, kw_counts


# --- 3. Main driver ---
def main():
    df = pd.read_csv(CLEAN_FILE)
    records = []

    print(f"Extracting sentiment and keyword features from {len(df)} sections...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["content"]).lower()
        sentiment, kw_counts = extract_features(text)

        record = {
            "filename": row["filename"],
            "year": row["year"],
            "month_name": row["month_name"],
            "section": row["section"],
            "sentiment": sentiment,
        }
        record.update(kw_counts)
        records.append(record)

    feat_df = pd.DataFrame(records)

    # Aggregate by report (average sentiment + total keyword counts)
    report_df = (
        feat_df.groupby(["year", "month_name"], as_index=False)
        .agg(
            {
                "sentiment": "mean",
                **{k: "sum" for k in KEYWORDS.keys()},
            }
        )
        .sort_values(["year", "month_name"])
    )

    feat_df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nExtracted features saved -> {OUT_FILE}")
    print(f"Total rows: {len(feat_df)}")
    print("\nAggregated report-level features preview:")
    print(report_df.head())


if __name__ == "__main__":
    main()
