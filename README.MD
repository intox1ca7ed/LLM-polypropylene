# LLM-polypropylene

> How much do polypropylene (PP) prices follow oil – and can OPEC text sentiment help us forecast PP moves?

This project combines **market data** and **LLM-based text analysis** to study European polypropylene (PP) prices.  
We:

- build clean monthly time series for **PP, feedstocks (PGP), and energy benchmarks (Brent, WTI, NatGas)**  
- extract **section-level sentiment** from **OPEC Monthly Oil Market Reports (MOMR)** with **FinBERT** and **GPT-4.1**  
- create a **hybrid sentiment index** (demand / supply / price-outlook) from 2019–2025  
- test whether that sentiment helps **predict PP prices** beyond crude and PGP

Spoiler: crude and PGP do most of the heavy lifting; OPEC sentiment adds, at best, **modest incremental information** – which is still an interesting result.

---

## Repository structure

```text
LLM-polypropylene/
├─ data/           # Raw and intermediate CSVs (prices, OPEC text, bvse plastics reports, etc.)
├─ notebooks/      # 01–08: EDA, baselines, sentiment index, modeling
├─ src/            # Reusable Python modules (EDA, sentiment, modeling utils)
├─ artifacts/      # Cleaned datasets, model-ready tables, metrics
├─ plots/          # PNGs used in the report / slides
└─ scripts/        # One-off helpers (scraping, preprocessing)