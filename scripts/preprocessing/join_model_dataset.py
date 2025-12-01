"""
Join GPT comparison sentiment, hybrid index, FinBERT features, and price data
into a single modeling-ready dataset.
"""

from pathlib import Path
import sys

import pandas as pd

# ---------------------
# Base directories
# ---------------------
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.llm.sentiment_hybrid_baseline import build_hybrid_sentiment

DATA_DIR = BASE_DIR / "data"
OPEC_DIR = DATA_DIR / "reports" / "energy" / "opec"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

HYBRID_FILE = OPEC_DIR / "opec_hybrid_sentiment.csv"
FINBERT_FILE = OPEC_DIR / "opec_features_finbert_chunked.csv"
PRICE_FILE = PROCESSED_DIR / "master_monthly_prices.csv"

OUT_FILE = PROCESSED_DIR / "master_opec_price_model_dataset.csv"


def load_hybrid_sentiment():
    """Load hybrid sentiment (ensures baseline row and month-end alignment)."""
    # Always (re)build to guarantee baseline and continuity
    hybrid = build_hybrid_sentiment(data_dir=OPEC_DIR, output_path=HYBRID_FILE)
    hybrid["date"] = pd.to_datetime(hybrid["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    hybrid["comparison_score"] = pd.to_numeric(hybrid["comparison_score"], errors="coerce").fillna(0.0)
    hybrid["hybrid_index"] = pd.to_numeric(hybrid["hybrid_index"], errors="coerce").fillna(0.0)

    # Guarantee full monthly coverage between min/max dates
    all_months = pd.date_range(hybrid["date"].min(), hybrid["date"].max(), freq="ME")
    hybrid = (
        hybrid.set_index("date")
        .reindex(all_months)
        .rename_axis("date")
        .reset_index()
        .fillna({"comparison_score": 0.0})
    )
    hybrid["hybrid_index"] = hybrid["comparison_score"].cumsum()
    return hybrid


def load_finbert_features():
    fin = pd.read_csv(FINBERT_FILE)
    fin["date"] = pd.to_datetime(
        fin["year"].astype(str) + "-" + fin["month_name"].astype(str) + "-01", errors="coerce"
    )
    fin["date"] = fin["date"] + pd.offsets.MonthEnd(0)

    numeric_cols = [
        c
        for c in fin.columns
        if c not in {"filename", "year", "month_name", "section", "date"}
    ]

    agg_fin = (
        fin.groupby("date", as_index=False)[numeric_cols]
        .mean()
        .sort_values("date")
    )
    return agg_fin


def load_prices():
    prices = pd.read_csv(PRICE_FILE)
    prices["date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.drop(columns=["Date"])
    return prices.sort_values("date")


def main():
    hybrid = load_hybrid_sentiment()
    fin = load_finbert_features()
    prices = load_prices()

    print("Merging FinBERT features with hybrid sentiment...")
    df = pd.merge(fin, hybrid, on="date", how="left")

    print("Adding monthly price data...")
    df = pd.merge(df, prices, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)

    # Create NEXT-MONTH targets
    df["PP_EU_next_month"] = df["PP_EU"].shift(-1)
    df["Brent_next_month"] = df["Brent"].shift(-1)

    # Drop trailing rows without targets (e.g., beyond price coverage)
    df = df.dropna(subset=["PP_EU", "Brent", "WTI", "NatGas", "PP_EU_next_month", "Brent_next_month"])

    # Select important columns
    keyword_cols = [c for c in df.columns if any(k in c for k in ["supply_", "demand_", "price_"])]
    cols_to_keep = [
        "date",
        "comparison_score",
        "hybrid_index",
        "finbert_sentiment",
    ] + keyword_cols + [
        "PP_EU",
        "Brent",
        "WTI",
        "NatGas",
        "PP_EU_next_month",
        "Brent_next_month",
    ]

    final_df = df[cols_to_keep]
    final_df.to_csv(OUT_FILE, index=False)

    print("\nMASTER MODELING DATASET SAVED:")
    print(OUT_FILE)
    print("\nPreview:")
    print(final_df.head(12))
    print(f"\nRows: {len(final_df)}, date range: {final_df['date'].min().date()} -> {final_df['date'].max().date()}")


if __name__ == "__main__":
    main()
