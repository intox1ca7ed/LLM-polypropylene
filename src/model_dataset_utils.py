"""
Utilities for assembling a monthly modeling dataset that combines PP prices,
returns, residuals, and OPEC sentiment indices. All inputs come from existing
artifacts; no external APIs are called.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .opec_sentiment_utils import build_opec_sentiment_index
except ImportError:
    # Allow execution as a standalone script by appending repo root to sys.path.
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from src.opec_sentiment_utils import build_opec_sentiment_index


@dataclass
class ModelingArtifacts:
    """Container for loaded price/residual artifacts."""

    monthly_prices: pd.DataFrame
    monthly_returns: pd.DataFrame
    residuals: pd.DataFrame


@dataclass
class OpecIndexArtifacts:
    """Container for loaded OPEC sentiment index artifacts."""

    monthly_index: pd.DataFrame
    monthly_section_scores: pd.DataFrame


def _read_csv_monthly(path: Path, date_col: str | None = None) -> pd.DataFrame:
    """
    Read a CSV with a monthly date index or date column and align to month-end.
    """
    df = pd.read_csv(path, parse_dates=[date_col] if date_col else True, index_col=0 if date_col is None else None)
    if date_col and date_col in df.columns:
        df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index = df.index + pd.offsets.MonthEnd(0)
    return df.sort_index()


def _find_first_matching(directory: Path, patterns: Sequence[str]) -> Optional[Path]:
    """
    Return the first file in directory matching all provided substrings.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if all(p.lower() in name for p in patterns):
            return path
    return None


def load_price_artifacts(root_path: Path) -> ModelingArtifacts:
    """
    Load monthly prices, returns, and PP residuals from artifacts/.
    Prints basic coverage and column availability.
    """
    root_path = Path(root_path)
    art = root_path / "artifacts"

    prices_path = (art / "merged_monthly_prices.csv") if (art / "merged_monthly_prices.csv").exists() else _find_first_matching(art, ["merged", "price"])
    returns_path = (art / "merged_monthly_returns.csv") if (art / "merged_monthly_returns.csv").exists() else _find_first_matching(art, ["merged", "return"])
    residual_path = (art / "pp_idiosyncratic_residual.csv") if (art / "pp_idiosyncratic_residual.csv").exists() else _find_first_matching(art, ["residual"])

    monthly_prices = _read_csv_monthly(prices_path) if prices_path else pd.DataFrame()
    monthly_returns = _read_csv_monthly(returns_path) if returns_path else pd.DataFrame()
    residuals = _read_csv_monthly(residual_path) if residual_path else pd.DataFrame()

    if not monthly_prices.empty:
        print(f"Loaded monthly prices from {prices_path}, columns: {list(monthly_prices.columns)}, coverage: {monthly_prices.index.min().date()} to {monthly_prices.index.max().date()}")
    else:
        print("Monthly prices not found.")

    if not monthly_returns.empty:
        print(f"Loaded monthly returns from {returns_path}, columns: {list(monthly_returns.columns)}, coverage: {monthly_returns.index.min().date()} to {monthly_returns.index.max().date()}")
    else:
        print("Monthly returns not found.")

    if not residuals.empty:
        print(f"Loaded residuals from {residual_path}, rows: {len(residuals)}, coverage: {residuals.index.min().date()} to {residuals.index.max().date()}")
    else:
        print("PP residuals not found.")

    return ModelingArtifacts(monthly_prices=monthly_prices, monthly_returns=monthly_returns, residuals=residuals)


def load_opec_index_artifacts(root_path: Path) -> OpecIndexArtifacts:
    """
    Load OPEC monthly index artifacts produced by notebook 06. If missing,
    recompute in-memory using build_opec_sentiment_index.
    """
    root_path = Path(root_path)
    art = root_path / "artifacts"

    idx_path = art / "opec_sentiment_monthly_index.csv"
    scores_path = art / "opec_sentiment_monthly_section_scores.csv"

    monthly_index = pd.DataFrame()
    monthly_scores = pd.DataFrame()

    if idx_path.exists():
        idx_df = pd.read_csv(idx_path)
        date_col = "report_date" if "report_date" in idx_df.columns else idx_df.columns[0]
        monthly_index = _read_csv_monthly(idx_path, date_col=date_col)
        print(f"Loaded OPEC monthly index from {idx_path}, columns: {list(monthly_index.columns)}")
    else:
        print("OPEC monthly index CSV not found; recomputing in memory.")
        built = build_opec_sentiment_index(root_path)
        monthly_index = built.get("monthly_index", pd.DataFrame())
        monthly_scores = built.get("monthly_section_scores", pd.DataFrame())

    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        date_col = "report_date" if "report_date" in scores_df.columns else scores_df.columns[0]
        monthly_scores = _read_csv_monthly(scores_path, date_col=date_col)
        print(f"Loaded OPEC section scores from {scores_path}")
    elif monthly_scores.empty and not monthly_index.empty:
        # Try to derive section scores from monthly_index score_* columns if present
        score_cols = [c for c in monthly_index.columns if c.startswith("score_")]
        if score_cols:
            monthly_scores = monthly_index[score_cols].copy()

    if not monthly_index.empty:
        cov = (monthly_index.index.min().date(), monthly_index.index.max().date())
        print(f"OPEC index coverage: {cov[0]} to {cov[1]}")
    else:
        print("OPEC index data not available.")

    return OpecIndexArtifacts(monthly_index=monthly_index, monthly_section_scores=monthly_scores)


def _ensure_month_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce") + pd.offsets.MonthEnd(0)
    out = out[~out.index.isna()]
    return out.sort_index()


def build_modeling_dataset(root_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Build the merged monthly dataset for modeling.
    Returns (DataFrame, metadata dict).
    """
    prices = load_price_artifacts(root_path)
    opec = load_opec_index_artifacts(root_path)

    frames = [prices.monthly_returns, prices.monthly_prices, prices.residuals, opec.monthly_index]
    frames = [_ensure_month_index(f) for f in frames if isinstance(f, pd.DataFrame) and not f.empty]

    if not frames:
        print("No data available to build the modeling dataset.")
        return pd.DataFrame(), {}

    combined_index = frames[0].index
    for f in frames[1:]:
        combined_index = combined_index.union(f.index)

    df_model = pd.DataFrame(index=combined_index.sort_values())

    # Prices
    if not prices.monthly_prices.empty:
        for col in prices.monthly_prices.columns:
            df_model[col] = prices.monthly_prices.reindex(df_model.index)[col]

    # Returns
    if not prices.monthly_returns.empty:
        for col in prices.monthly_returns.columns:
            df_model[f"ret_{col}"] = prices.monthly_returns.reindex(df_model.index)[col]

    # Residuals
    if not prices.residuals.empty:
        resid_col = prices.residuals.columns[0] if not prices.residuals.columns.empty else "pp_residual"
        df_model["resid_PP"] = prices.residuals.reindex(df_model.index)[resid_col]

    # OPEC sentiment indices
    if not opec.monthly_index.empty:
        for col in opec.monthly_index.columns:
            if col.startswith("index_") or col.startswith("score_"):
                df_model[col] = opec.monthly_index.reindex(df_model.index)[col]

    # Drop rows where all core modeling columns are missing
    core_cols = [c for c in ["ret_PP", "resid_PP", "index_hybrid"] if c in df_model.columns]
    if core_cols:
        df_model = df_model.dropna(subset=core_cols, how="all")

    metadata = {
        "start_date": str(df_model.index.min().date()) if not df_model.empty else "",
        "end_date": str(df_model.index.max().date()) if not df_model.empty else "",
        "n_months": str(len(df_model)),
        "columns": ", ".join(df_model.columns),
    }

    print(f"Modeling dataset shape: {df_model.shape}")
    print(f"Coverage: {metadata['start_date']} to {metadata['end_date']}")
    return df_model, metadata


def summarize_modeling_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Produce descriptive stats, missing counts, and a correlation matrix for key variables.
    """
    if df.empty:
        return {"describe": pd.DataFrame(), "missing": pd.DataFrame(), "corr": pd.DataFrame()}

    numeric_cols = df.select_dtypes(include=["number"]).columns
    desc = df[numeric_cols].describe().T
    missing = df[numeric_cols].isna().sum().to_frame(name="missing_count")

    corr_cols = [c for c in ["ret_PP", "resid_PP", "index_hybrid", "index_demand", "index_supply", "index_price_outlook", "ret_CRUDE", "ret_PGP"] if c in df.columns]
    corr = df[corr_cols].corr() if corr_cols else pd.DataFrame()

    return {"describe": desc, "missing": missing, "corr": corr}


def plot_pp_vs_opec_index(df: pd.DataFrame) -> plt.Figure:
    """
    Plot PP returns or residuals alongside the OPEC hybrid index over time.
    """
    if df.empty or "index_hybrid" not in df.columns:
        raise ValueError("Data or index_hybrid column is missing.")

    target_col = "resid_PP" if "resid_PP" in df.columns else "ret_PP"
    if target_col not in df.columns:
        raise ValueError("Neither resid_PP nor ret_PP is available to plot.")

    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=120)
    ax1.plot(df.index, df[target_col], color="steelblue", label=target_col)
    ax1.set_ylabel(target_col)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["index_hybrid"], color="darkred", label="index_hybrid")
    ax2.set_ylabel("index_hybrid")

    ax1.set_title(f"{target_col} vs OPEC hybrid index")
    ax1.set_xlabel("Month")
    ax1.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.autofmt_xdate()
    return fig


def plot_residual_vs_index_scatter(df: pd.DataFrame, section: str | None = None) -> plt.Figure:
    """
    Scatter plot of PP residuals (or returns) vs hybrid or section index with a regression line.
    """
    if df.empty:
        raise ValueError("Dataframe is empty.")

    y_col = "resid_PP" if "resid_PP" in df.columns else "ret_PP"
    if y_col not in df.columns:
        raise ValueError("Neither resid_PP nor ret_PP is available for plotting.")

    x_col = "index_hybrid"
    if section:
        candidate = f"index_{section.lower()}"
        if candidate in df.columns:
            x_col = candidate

    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        raise ValueError("No overlapping data to plot.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sns.regplot(x=x_col, y=y_col, data=plot_df, ax=ax, scatter_kws={"alpha": 0.6, "s": 30}, line_kws={"color": "darkred"})
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.3)
    return fig


def plot_leadlag_sentiment_vs_residual(df: pd.DataFrame, max_lag: int = 6) -> plt.Figure:
    """
    Bar plot of correlations between PP residuals (or returns) and index_hybrid
    across lags from -max_lag to +max_lag. Positive lag means sentiment leads.
    """
    if df.empty or "index_hybrid" not in df.columns:
        raise ValueError("Data or index_hybrid column is missing.")

    target_col = "resid_PP" if "resid_PP" in df.columns else "ret_PP"
    if target_col not in df.columns:
        raise ValueError("Neither resid_PP nor ret_PP is available.")

    base = df[[target_col, "index_hybrid"]].dropna()
    if base.empty:
        raise ValueError("Insufficient overlapping data for lead/lag correlations.")

    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        shifted = base["index_hybrid"].shift(lag)
        corrs.append(base[target_col].corr(shifted))

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar(lags, corrs, color="steelblue", alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (months, positive = sentiment leads)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Lead/Lag correlations: {target_col} vs index_hybrid")
    ax.grid(True, alpha=0.3)
    return fig
