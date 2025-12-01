from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PRICE_DIR = BASE_DIR / "data" / "prices"
PRICE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PRICE_DIR / "crude_oil_weekly.csv"
OUTPUT_PATH = PRICE_DIR / "crude_oil_weekly_clean.csv"


def main():
    df = pd.read_csv(INPUT_PATH)
    print("Original columns:", df.columns.tolist())

    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(".", "", regex=False)

    wti = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    brent = df[["Date", "Open1", "High1", "Low1", "Close1", "Volume1"]].copy()

    wti.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    brent.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    wti["Commodity"] = "WTI_Crude"
    brent["Commodity"] = "Brent_Crude"

    combined = pd.concat([wti, brent], ignore_index=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        combined = combined[~combined[col].astype(str).str.contains("CL=F|BZ=F|Brent|WTI|NaN", case=False, na=False)]

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        combined[col] = (
            combined[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    combined = combined[["Date", "Commodity", "Open", "High", "Low", "Close", "Volume"]]
    combined = combined.dropna(subset=["Date"]).sort_values(["Commodity", "Date"]).reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned crude oil weekly data saved to {OUTPUT_PATH}")

    print("\nSummary:")
    print(combined.groupby("Commodity")["Date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
