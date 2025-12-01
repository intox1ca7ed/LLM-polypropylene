"""
Prepare master monthly dataset by combining polypropylene prices,
crude (Brent/WTI) weekly data, and natural gas weekly data.
"""

from pathlib import Path

import pandas as pd

# -------------------------------
# 1. Define base paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
PRICE_DIR = BASE_DIR / "data" / "prices"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_polypropylene():
    pp = pd.read_csv(PRICE_DIR / "polypropylene_primary_avg_prices.csv")
    pp["Date"] = pd.to_datetime(pp["Month"], format="%b %Y") + pd.offsets.MonthEnd(0)
    pp = pp.rename(columns={"PP_Avg_EUR_per_t": "PP_EU"})
    pp = pp[["Date", "PP_EU"]].set_index("Date").sort_index()
    return pp.resample("M").mean()


def load_crude():
    oil = pd.read_csv(PRICE_DIR / "crude_oil_weekly_clean.csv", parse_dates=["Date"])
    brent = (
        oil[oil["Commodity"] == "Brent_Crude"][["Date", "Close"]]
        .set_index("Date")
        .resample("M")
        .mean()
        .rename(columns={"Close": "Brent"})
    )
    wti = (
        oil[oil["Commodity"] == "WTI_Crude"][["Date", "Close"]]
        .set_index("Date")
        .resample("M")
        .mean()
        .rename(columns={"Close": "WTI"})
    )
    return pd.concat([brent, wti], axis=1)


def load_natgas():
    gas = pd.read_csv(PRICE_DIR / "natgas_weekly.csv", parse_dates=["Date"])
    gas = gas[["Date", "Close"]].copy()
    gas["Close"] = pd.to_numeric(gas["Close"], errors="coerce")
    gas = gas.rename(columns={"Close": "NatGas"})
    return gas.set_index("Date").resample("M").mean()


def main():
    pp_m = load_polypropylene()
    oil_m = load_crude()
    gas_m = load_natgas()

    df = pd.concat([pp_m, oil_m, gas_m], axis=1).sort_index()
    df = df[df.index >= "2019-01-01"]

    # Fill small gaps, drop remaining NaNs
    df = df.ffill().dropna(how="any")
    df = df.round(2)

    print("\nCombined dataset preview:")
    print(df.head(10))
    print("\nData coverage:", df.index.min().strftime("%b %Y"), "->", df.index.max().strftime("%b %Y"))
    print("\nColumns:", list(df.columns))
    print("\nDescription:\n", df.describe())

    output_path = PROCESSED_DIR / "master_monthly_prices.csv"
    df.to_csv(output_path, index=True)
    print(f"\nMaster monthly dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
