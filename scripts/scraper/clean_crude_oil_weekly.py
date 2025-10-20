import pandas as pd
import os

# --- Setup ---
input_path = "data/prices/crude_oil_weekly.csv"
output_path = "data/prices/crude_oil_weekly_clean.csv"

# Ensure folder exists
os.makedirs("data/prices", exist_ok=True)

# --- Load Data ---
df = pd.read_csv(input_path)
print("Original columns:", df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(".", "", regex=False)

# --- Split into WTI and Brent ---
wti = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
brent = df[["Date", "Open1", "High1", "Low1", "Close1", "Volume1"]].copy()

# Fix column names
wti.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
brent.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Add commodity names
wti["Commodity"] = "WTI_Crude"
brent["Commodity"] = "Brent_Crude"

# --- Combine ---
combined = pd.concat([wti, brent], ignore_index=True)

# --- Clean Data ---
# Remove rows that contain text like 'CL=F' or 'BZ=F' instead of numbers
for col in ["Open", "High", "Low", "Close", "Volume"]:
    combined = combined[~combined[col].astype(str).str.contains("CL=F|BZ=F|Brent|WTI|NaN", case=False, na=False)]

# Convert date and numeric values
combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")

numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in numeric_cols:
    combined[col] = (
        combined[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

# --- Sort & Save ---
combined = combined[["Date", "Commodity", "Open", "High", "Low", "Close", "Volume"]]
combined = combined.dropna(subset=["Date"]).sort_values(["Commodity", "Date"]).reset_index(drop=True)

# --- Save Clean File ---
combined.to_csv(output_path, index=False)
print(f"✅ Cleaned crude oil weekly data saved to {output_path}")

# Optional summary
print("\nSummary:")
print(combined.groupby("Commodity")["Date"].agg(["min", "max", "count"]))
