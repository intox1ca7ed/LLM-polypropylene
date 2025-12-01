import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[2]
PRICE_DIR = BASE_DIR / "data" / "prices"
PRICE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    tickers = {
        "WTI_Crude": "CL=F",
        "Brent_Crude": "BZ=F",
    }

    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    data_frames = []
    for name, symbol in tickers.items():
        print(f"Fetching data for {name} ({symbol})...")
        df = yf.download(symbol, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df["Commodity"] = name
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df = combined_df[["Date", "Commodity", "Open", "High", "Low", "Close", "Volume"]]

    daily_path = PRICE_DIR / "crude_oil_daily.csv"
    combined_df.to_csv(daily_path, index=False)
    print(f"Daily crude oil prices saved to {daily_path}")

    weekly_df = (
        combined_df.set_index("Date")
        .groupby("Commodity")
        .resample("W-Mon")
        .mean(numeric_only=True)
        .reset_index()
    )

    weekly_path = PRICE_DIR / "crude_oil_weekly.csv"
    weekly_df.to_csv(weekly_path, index=False)
    print(f"Weekly crude oil prices saved to {weekly_path}")


if __name__ == "__main__":
    main()
