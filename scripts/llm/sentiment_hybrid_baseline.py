"""
Build a stable hybrid sentiment index from GPT comparison scores.
Baseline: Jan 2019 comparison_score = 0, then cumulative month-to-month deltas.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OPEC_DIR = BASE_DIR / "data" / "reports" / "energy" / "opec"
GPT_PATH = OPEC_DIR / "opec_comparison_scores_gpt.csv"
OUT_PATH = OPEC_DIR / "opec_hybrid_sentiment.csv"


def build_hybrid_sentiment(data_dir: Path = OPEC_DIR, output_path: Path | None = None) -> pd.DataFrame:
    """Load GPT comparison scores, add Jan-2019 baseline, and build hybrid index."""
    data_dir = Path(data_dir)
    gpt_path = data_dir / "opec_comparison_scores_gpt.csv"
    output_path = Path(output_path) if output_path else data_dir / "opec_hybrid_sentiment.csv"

    df = pd.read_csv(gpt_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["comparison_score"] = pd.to_numeric(df["comparison_score"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    baseline_date = pd.Timestamp("2019-01-01")
    baseline_row = pd.DataFrame([{"date": baseline_date, "comparison_score": 0.0}])
    df = pd.concat([baseline_row, df], ignore_index=True)
    df = df.drop_duplicates(subset="date", keep="first")

    # Align to month-end and ensure monthly continuity
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    all_months = pd.date_range(df["date"].min(), df["date"].max(), freq="ME")
    df = (
        df[["date", "comparison_score"]]
        .set_index("date")
        .reindex(all_months)
        .rename_axis("date")
        .reset_index()
    )

    df["comparison_score"] = df["comparison_score"].fillna(0.0)
    df["hybrid_index"] = df["comparison_score"].cumsum()

    df.to_csv(output_path, index=False)
    return df[["date", "comparison_score", "hybrid_index"]]


def main():
    hybrid = build_hybrid_sentiment()
    print("Hybrid sentiment saved to:", OUT_PATH)
    print("\nPreview:\n", hybrid.head())
    jan_2019 = hybrid.loc[hybrid["date"] == pd.Timestamp("2019-01-31"), "hybrid_index"]
    print("\nJan 2019 baseline hybrid_index:", jan_2019.iloc[0] if not jan_2019.empty else "missing")


if __name__ == "__main__":
    main()
