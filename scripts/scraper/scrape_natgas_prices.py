from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[2]
PRICE_DIR = BASE_DIR / "data" / "prices"
PRICE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("Fetching natural gas price data...")
    df = yf.download("NG=F", start=start_date, end=end_date)
    df.reset_index(inplace=True)

    daily_path = PRICE_DIR / "natgas_daily.csv"
    df.to_csv(daily_path, index=False)
    print(f"Daily natural gas prices saved to {daily_path}")

    weekly_df = df.resample("W-Mon", on="Date").mean(numeric_only=True).reset_index()
    weekly_path = PRICE_DIR / "natgas_weekly.csv"
    weekly_df.to_csv(weekly_path, index=False)
    print(f"Weekly natural gas prices saved to {weekly_path}")


if __name__ == "__main__":
    main()
