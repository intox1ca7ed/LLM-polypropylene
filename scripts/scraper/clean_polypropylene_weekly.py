import pandas as pd
import os

# Load the CSV file
file_path = "data/prices/polypropylene_weekly.csv"

# Read data
df = pd.read_csv(file_path)

print("Original columns:", df.columns)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace('.', '', regex=False)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Sort by date
df = df.sort_values('date')

# If price is stored with commas (e.g., '1,230'), remove them
df['price'] = df['price'].astype(str).str.replace(',', '').astype(float)

# Add commodity name
df['commodity'] = 'Polypropylene_Futures'

# Keep only the relevant columns
df = df[['date', 'commodity', 'price']]

# Save clean version
output_path = "data/prices/polypropylene_weekly_clean.csv"
df.to_csv(output_path, index=False)
print(f" Cleaned polypropylene weekly prices saved to {output_path}")
