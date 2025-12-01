from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PRICE_DIR = BASE_DIR / "data" / "prices"
PRICE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PRICE_DIR / "polypropylene_weekly.csv"
OUTPUT_PATH = PRICE_DIR / "polypropylene_weekly_clean.csv"


def main():
    df = pd.read_csv(INPUT_PATH)
    print("Original columns:", df.columns)

    df.columns = df.columns.str.strip().str.lower().str.replace(".", "", regex=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    df["price"] = df["price"].astype(str).str.replace(",", "", regex=False).astype(float)
    df["commodity"] = "Polypropylene_Futures"

    df = df[["date", "commodity", "price"]]
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned polypropylene weekly prices saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
