import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Ensure output folder exists
os.makedirs('data/prices', exist_ok=True)

# Define ticker symbols and names
tickers = {
    "WTI_Crude": "CL=F",   # WTI Crude Oil
    "Brent_Crude": "BZ=F"  # Brent Crude Oil
}

# Date range
start_date = "2015-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

data_frames = []

# Fetch data
for name, symbol in tickers.items():
    print(f"Fetching data for {name} ({symbol})...")
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df["Commodity"] = name
    data_frames.append(df)

# Combine
combined_df = pd.concat(data_frames, ignore_index=True)

# Keep relevant columns (exclude Adj Close)
combined_df = combined_df[["Date", "Commodity", "Open", "High", "Low", "Close", "Volume"]]

# Save daily data
combined_df.to_csv('data/prices/crude_oil_daily.csv', index=False)
print("Daily crude oil prices saved to data/prices/crude_oil_daily.csv")

# Convert to weekly averages
weekly_df = (
    combined_df
    .set_index("Date")
    .groupby("Commodity")
    .resample("W-Mon")  # weekly ending Monday
    .mean(numeric_only=True)
    .reset_index()
)

weekly_df.to_csv('data/prices/crude_oil_weekly.csv', index=False)
print("Weekly crude oil prices saved to data/prices/crude_oil_weekly.csv")
