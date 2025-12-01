import re
from pathlib import Path

import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
OPEC_DIR = BASE_DIR / "data" / "reports" / "energy" / "opec"
CLEAN_FILE = OPEC_DIR / "opec_texts_clean_sections.csv"
OUT_FILE = OPEC_DIR / "opec_features_finbert.csv"

# --- Load FinBERT ---
MODEL = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# --- Keywords ---
KEYWORDS = {
    "supply_up": ["increase", "rise", "growth", "expand", "surge"],
    "supply_down": ["decrease", "cut", "decline", "drop", "reduce", "slowdown"],
    "demand_up": ["demand growth", "strong demand", "recover", "rise"],
    "demand_down": ["weak demand", "sluggish", "decline", "fall", "drop"],
    "price_up": ["higher price", "price increase", "upward trend", "strengthened"],
    "price_down": ["lower price", "price fall", "downward", "weakened"],
}


def keyword_density(text, word_list):
    """Compute normalized keyword count."""
    if not text:
        return 0
    words = len(text.split())
    count = 0
    for w in word_list:
        count += len(re.findall(r"\b" + re.escape(w) + r"\b", text, flags=re.I))
    return round(count / max(words, 1), 5)


def finbert_sentiment(text):
    """Return FinBERT sentiment polarity: positive minus negative."""
    try:
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = softmax(outputs.logits.numpy())[0]
        return float(scores[2] - scores[0])  # positive - negative
    except Exception:
        return 0.0


def main():
    df = pd.read_csv(CLEAN_FILE)
    print(f"Processing {len(df)} OPEC sections with FinBERT...")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["content"])
        section = row["section"]

        # FinBERT sentiment
        finbert_score = finbert_sentiment(text)

        # Keyword densities
        densities = {k: keyword_density(text, v) for k, v in KEYWORDS.items()}

        rec = {
            "filename": row["filename"],
            "year": row["year"],
            "month_name": row["month_name"],
            "section": section,
            "finbert_sentiment": finbert_score,
        }
        rec.update(densities)
        records.append(rec)

    features_df = pd.DataFrame(records)

    # Aggregate by report: average sentiment + mean keyword density
    report_df = (
        features_df.groupby(["year", "month_name"], as_index=False)
        .agg(
            {
                "finbert_sentiment": "mean",
                **{k: "mean" for k in KEYWORDS.keys()},
            }
        )
        .sort_values(["year", "month_name"])
    )

    features_df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nFinBERT features saved -> {OUT_FILE}")
    print(report_df.head())


if __name__ == "__main__":
    main()
