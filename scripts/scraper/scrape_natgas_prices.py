import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Ensure output folder exists
os.makedirs('data/prices', exist_ok=True)

# Define ticker symbols and names
tickers = {
    "NG=F"
}

# Date range    
start_date = "2015-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print("Fetching natural gas price data...")
df = yf.download("NG=F", start=start_date, end=end_date)
df.reset_index(inplace=True)

#save daily data
df.to_csv('data/prices/natgas_daily.csv', index=False)
print("Daily natural gas prices saved to data/prices/natgas_daily.csv")

# Convert to weekly averages
weekly_df = (df.resample("W-Mon", on="Date").mean(numeric_only=True).reset_index())
weekly_df.to_csv('data/prices/natgas_weekly.csv', index=False)
print("Weekly natural gas prices saved to data/prices/natgas_weekly.csv")
