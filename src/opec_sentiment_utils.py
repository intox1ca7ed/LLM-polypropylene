"""
Utilities for loading OPEC sentiment artifacts and building a chained index.

Relies only on locally saved outputs (FinBERT features, GPT comparisons) and
does not call any external APIs. The main entry point is
``build_opec_sentiment_index`` which discovers files, loads them, aggregates
monthly section scores, chains them from a baseline month, and provides simple
plotting helpers.
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_BASELINE_DATE = "2019-01-01"
_SEARCH_KEYWORDS = ("opec", "momr", "sentiment", "finbert", "gpt")

_MONTH_LOOKUP = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTH_LOOKUP.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def find_opec_sentiment_files(root_path: Path) -> Dict[str, List[Path]]:
    """
    Discover OPEC sentiment-related files under data/ and artifacts/.

    Files whose names contain any of the keywords (opec, momr, sentiment,
    finbert, gpt) are grouped into categories: finbert, gpt, other.
    """
    root_path = Path(root_path)
    candidates: Dict[str, List[Path]] = {"finbert": [], "gpt": [], "other": []}
    search_roots = [root_path / "data", root_path / "artifacts"]

    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not any(k in name for k in _SEARCH_KEYWORDS):
                continue
            if "finbert" in name:
                candidates["finbert"].append(path)
            elif "gpt" in name:
                candidates["gpt"].append(path)
            else:
                candidates["other"].append(path)

    # Drop empty categories for cleanliness
    candidates = {k: sorted(v) for k, v in candidates.items() if v}

    if not candidates:
        print("No OPEC sentiment files found under data/ or artifacts/.")
    else:
        for cat, files in candidates.items():
            print(f"Discovered {len(files)} {cat} file(s).")
    return candidates


def _infer_date_from_filename(name: str) -> pd.Timestamp | pd.NaT:
    """Infer a month-end date from filename tokens such as 2019_01 or January_2019."""
    lower = name.lower()

    # Numeric year-month
    num_match = re.search(r"(20\\d{2})[._-]?([01]?\\d)", lower)
    if num_match:
        year, month = int(num_match.group(1)), int(num_match.group(2))
        try:
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        except ValueError:
            pass

    # Year + month name
    year_match = re.search(r"(20\\d{2})", lower)
    if year_match:
        year = int(year_match.group(1))
        for mon_name, mon_num in _MONTH_LOOKUP.items():
            if mon_name and mon_name in lower:
                try:
                    return pd.Timestamp(year=year, month=mon_num, day=1) + pd.offsets.MonthEnd(0)
                except ValueError:
                    continue
    return pd.NaT


def _infer_report_date(row: pd.Series, source_file: Path) -> pd.Timestamp | pd.NaT:
    """Infer report_date from common columns or the source filename."""
    for col in ("report_date", "date"):
        if col in row and pd.notna(row[col]):
            ts = pd.to_datetime(row[col], errors="coerce")
            if pd.notna(ts):
                return ts + pd.offsets.MonthEnd(0)

    year = row.get("year")
    month = row.get("month") or row.get("month_num")
    month_name = row.get("month_name")

    if pd.notna(year):
        month_val: int | None = None
        if pd.notna(month):
            try:
                month_val = int(month)
            except (TypeError, ValueError):
                month_val = None
        if month_val is None and pd.notna(month_name):
            month_val = _MONTH_LOOKUP.get(str(month_name).strip().lower())
        if month_val:
            try:
                return pd.Timestamp(year=int(year), month=month_val, day=1) + pd.offsets.MonthEnd(0)
            except ValueError:
                pass

    inferred = _infer_date_from_filename(source_file.name)
    return inferred


def _normalize_section(section: str) -> str:
    """Map noisy section labels to a small canonical set."""
    if not isinstance(section, str):
        return "other"
    s = section.strip().lower()
    if not s:
        return "other"
    if "hybrid" in s or "overall" in s:
        return "overall"
    if "demand" in s:
        return "demand"
    if "supply" in s:
        return "supply"
    if "balance" in s:
        return "balance"
    if "price" in s or "market" in s or "refining" in s or "outlook" in s:
        return "price_outlook"
    return "other"


def _load_finbert_file(path: Path) -> pd.DataFrame:
    """Standardise a FinBERT CSV into the canonical long format."""
    df = pd.read_csv(path)
    sentiment_col = None
    for cand in ("finbert_sentiment", "sentiment_finbert", "sentiment"):
        if cand in df.columns:
            sentiment_col = cand
            break

    out = pd.DataFrame()
    out["report_date"] = df.apply(lambda r: _infer_report_date(r, path), axis=1)
    out["section"] = df["section"] if "section" in df.columns else "overall"
    out["sentiment_finbert"] = (
        pd.to_numeric(df[sentiment_col], errors="coerce") if sentiment_col else np.nan
    )
    out["sentiment_gpt"] = np.nan
    out["source_file"] = str(path)
    return out


def _load_gpt_file(path: Path) -> pd.DataFrame:
    """Standardise a GPT comparison CSV into the canonical long format."""
    df = pd.read_csv(path)
    if "comparison_score" not in df.columns and "hybrid_index" in df.columns:
        df["comparison_score"] = pd.to_numeric(df["hybrid_index"], errors="coerce").diff().fillna(0.0)

    out = pd.DataFrame()
    out["report_date"] = df.apply(lambda r: _infer_report_date(r, path), axis=1)
    out["section"] = df["section"] if "section" in df.columns else "hybrid"
    out["sentiment_finbert"] = np.nan
    out["sentiment_gpt"] = pd.to_numeric(df.get("comparison_score"), errors="coerce")
    out["source_file"] = str(path)
    return out


def load_opec_sentiment_raw(files_by_type: Mapping[str, Sequence[Path]]) -> pd.DataFrame:
    """
    Load FinBERT and GPT sentiment files into a single long DataFrame.

    Returns a DataFrame with at least: report_date, section, sentiment_finbert,
    sentiment_gpt, source_file.
    """
    records: list[pd.DataFrame] = []
    loaded: list[str] = []

    for finbert_path in files_by_type.get("finbert", []):
        try:
            df = _load_finbert_file(finbert_path)
            records.append(df)
            loaded.append(f"FinBERT -> {finbert_path}")
        except Exception as exc:
            print(f"Skipping FinBERT file {finbert_path} due to error: {exc}")

    for gpt_path in files_by_type.get("gpt", []):
        try:
            df = _load_gpt_file(gpt_path)
            records.append(df)
            loaded.append(f"GPT -> {gpt_path}")
        except Exception as exc:
            print(f"Skipping GPT file {gpt_path} due to error: {exc}")

    if not records:
        print("No sentiment files could be loaded.")
        return pd.DataFrame(columns=["report_date", "section", "sentiment_finbert", "sentiment_gpt", "source_file"])

    raw = pd.concat(records, ignore_index=True)
    raw["report_date"] = pd.to_datetime(raw["report_date"], errors="coerce")
    raw = raw.dropna(subset=["report_date"]).sort_values("report_date")
    months_found = raw["report_date"].dt.to_period("M").nunique()

    print("Loaded sentiment files:")
    for line in loaded:
        print(f"  - {line}")
    print(f"Total rows: {len(raw)}, unique months: {months_found}")
    print("Report dates aligned to month-end using explicit date columns, year/month fields, or filename tokens.")

    return raw.reset_index(drop=True)


def build_hybrid_sentiment(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct monthly section scores using GPT where available, else FinBERT.

    Returns (monthly_section_scores, long_with_base) where monthly_section_scores
    is a wide DataFrame indexed by month-end dates.
    """
    if raw_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=["report_date", "section", "sentiment_finbert", "sentiment_gpt", "source_file", "base_sentiment", "section_clean"]),
        )

    df = raw_df.copy()
    df["base_sentiment"] = df["sentiment_gpt"].combine_first(df["sentiment_finbert"])
    df["section_clean"] = df["section"].apply(_normalize_section)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce") + pd.offsets.MonthEnd(0)
    df = df.dropna(subset=["report_date"])

    grouped = (
        df.groupby(["report_date", "section_clean"])["base_sentiment"]
        .mean()
        .unstack()
        .sort_index()
    )

    if not grouped.empty:
        full_months = pd.date_range(grouped.index.min(), grouped.index.max(), freq="ME")
        grouped = grouped.reindex(full_months)

    return grouped, df


def chain_sentiment_index(monthly_section_scores: pd.DataFrame, baseline_date: str = DEFAULT_BASELINE_DATE) -> pd.DataFrame:
    """
    Chain monthly section scores into cumulative indices starting from baseline.
    """
    if monthly_section_scores.empty:
        return pd.DataFrame()

    scores = monthly_section_scores.copy()
    scores.index = pd.to_datetime(scores.index) + pd.offsets.MonthEnd(0)
    scores = scores.sort_index()
    scores = scores.fillna(0.0)

    cumulative = scores.cumsum()
    baseline_ts = pd.Timestamp(baseline_date) + pd.offsets.MonthEnd(0)
    baseline_vector = cumulative.loc[baseline_ts] if baseline_ts in cumulative.index else cumulative.iloc[0]

    index_df = cumulative.sub(baseline_vector, axis=1)
    index_df.columns = [f"index_{c}" for c in index_df.columns]

    score_cols = {f"score_{c}": monthly_section_scores[c] for c in monthly_section_scores.columns}
    monthly_index = pd.concat([pd.DataFrame(score_cols), index_df], axis=1)

    section_index_cols = [c for c in monthly_index.columns if c.startswith("index_")]
    monthly_index["index_hybrid"] = monthly_index[section_index_cols].mean(axis=1, skipna=True)
    monthly_index = monthly_index.sort_index()

    return monthly_index


def build_opec_sentiment_index(root_path: Path) -> Dict[str, pd.DataFrame]:
    """
    High-level orchestrator that discovers, loads, aggregates, and chains the index.
    """
    files = find_opec_sentiment_files(root_path)
    raw = load_opec_sentiment_raw(files)

    monthly_section_scores, raw_enriched = build_hybrid_sentiment(raw)
    monthly_index = chain_sentiment_index(monthly_section_scores, baseline_date=DEFAULT_BASELINE_DATE)

    summary = {
        "raw": raw_enriched,
        "monthly_section_scores": monthly_section_scores,
        "monthly_index": monthly_index,
    }

    if raw.empty:
        print("No sentiment data available to build the index.")
    else:
        min_date = raw_enriched["report_date"].min()
        max_date = raw_enriched["report_date"].max()
        print(f"Index coverage: {min_date.date()} to {max_date.date()}")
    return summary


def _prepare_plot_index(monthly_index: pd.DataFrame) -> pd.DataFrame:
    if monthly_index.empty:
        raise ValueError("monthly_index is empty; nothing to plot.")
    df = monthly_index.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def plot_opec_index_sections(monthly_index: pd.DataFrame, sections: Iterable[str] | None = None) -> plt.Figure:
    """
    Plot section-level indices on a single figure.
    """
    df = _prepare_plot_index(monthly_index)
    section_cols = [c for c in df.columns if c.startswith("index_") and c != "index_hybrid"]
    if sections:
        desired = {f"index_{_normalize_section(s)}" for s in sections}
        section_cols = [c for c in section_cols if c in desired]
    if not section_cols:
        raise ValueError("No section index columns found to plot.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for col in section_cols:
        ax.plot(df.index, df[col], label=col.replace("index_", ""))
    ax.set_title("OPEC Sentiment Index by Section")
    ax.set_xlabel("Report Month")
    ax.set_ylabel("Cumulative sentiment vs baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_opec_index_hybrid(monthly_index: pd.DataFrame) -> plt.Figure:
    """
    Plot the aggregate hybrid index over time.
    """
    df = _prepare_plot_index(monthly_index)
    if "index_hybrid" not in df.columns:
        raise ValueError("index_hybrid column not found.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(df.index, df["index_hybrid"], color="black", label="Hybrid index")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("OPEC Hybrid Sentiment Index")
    ax.set_xlabel("Report Month")
    ax.set_ylabel("Cumulative sentiment vs baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_opec_index_changes(monthly_index: pd.DataFrame) -> plt.Figure:
    """
    Plot month-over-month changes in the hybrid index.
    """
    df = _prepare_plot_index(monthly_index)
    if "index_hybrid" not in df.columns:
        raise ValueError("index_hybrid column not found.")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    delta = df["index_hybrid"].diff()
    ax.bar(df.index, delta, color="steelblue", width=20, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Hybrid Index Month-over-Month Change")
    ax.set_xlabel("Report Month")
    ax.set_ylabel("Change vs prior month")
    fig.autofmt_xdate()
    return fig
