"""
Dry-run orchestrator that exercises eda_utils outside the notebook.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from eda_utils import (  # type: ignore
    build_audit_table,
    compute_log_returns,
    cross_corr_scan,
    find_candidate_files,
    load_series,
    normalize_base_100,
    plot_beta_scaled_spread,
    plot_leadlag_heatmap,
    plot_normalized_levels,
    plot_rolling_beta,
    plot_scatter_returns,
    rolling_beta_ols,
    rolling_corr,
    to_monthly,
)

DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_KEYWORDS: Dict[str, List[str]] = {
    "PP": ["pp", "polypropylene"],
    "PGP": ["pgp", "propylene", "c3"],
    "CRUDE": ["brent", "wti", "crude", "oil"],
    "NAPHTHA": ["naphtha"],
}


def main() -> None:
    discovery: Dict[str, List[Path]] = {
        label: find_candidate_files(DATA_DIR, keywords)
        for label, keywords in CLASS_KEYWORDS.items()
    }

    seen_paths: set[Path] = set()
    for label in CLASS_KEYWORDS:
        unique: List[Path] = []
        for path in discovery.get(label, []):
            if path in seen_paths:
                continue
            unique.append(path)
            seen_paths.add(path)
        discovery[label] = unique

    montly_series_per_class: Dict[str, Dict[str, pd.Series]] = {}
    for label, paths in discovery.items():
        series_collection: Dict[str, pd.Series] = {}
        for path in paths:
            try:
                df = load_series(path, series_label=f"{label}|{path.stem}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to load {path.name}: {exc}")
                continue
            monthly = to_monthly(df)
            series_collection[monthly.name] = monthly
            print(
                f"[LOAD] {path.name} -> {monthly.name}, rows {df.attrs['rows_clean']}/{df.attrs['rows_raw']}"
            )
        if series_collection:
            montly_series_per_class[label] = series_collection
        else:
            keywords = ", ".join(CLASS_KEYWORDS[label])
            print(f"[INFO] No series for {label}; expected keywords: {keywords}")

    if not montly_series_per_class:
        print("No monthly series discovered; exiting.")
        return

    wide_outer = pd.DataFrame()
    preferred: Dict[str, pd.Series] = {}
    for label, series_dict in montly_series_per_class.items():
        table = pd.concat(series_dict.values(), axis=1)
        wide_outer = pd.concat([wide_outer, table], axis=1)
        best = table.count().sort_values(ascending=False).index[0]
        preferred[label] = table[best]
    wide_outer = wide_outer.loc[~wide_outer.index.duplicated()].sort_index()
    print(f"[INFO] Monthly outer shape: {wide_outer.shape}")

    aligned = pd.concat(preferred, axis=1).dropna()
    print(f"[INFO] Monthly inner shape: {aligned.shape}")

    aligned.to_csv(ARTIFACTS_DIR / "merged_monthly_prices.csv", index=True)

    audit_input = {
        (series.name or label): series for label, series in preferred.items()
    }
    audit_table = build_audit_table(audit_input)
    audit_table.to_csv(ARTIFACTS_DIR / "data_audit.csv", index=False)
    print("[INFO] Saved data_audit.csv")

    norm_levels = normalize_base_100(aligned)
    fig = plot_normalized_levels(norm_levels, title="Normalized Levels (Base=100)")
    fig.savefig(PLOTS_DIR / "levels_normalized.png", bbox_inches="tight")
    plt.close(fig)

    returns = compute_log_returns(aligned).dropna()
    returns.to_csv(ARTIFACTS_DIR / "merged_monthly_returns.csv", index=True)

    if {"PP", "CRUDE"} <= set(aligned.columns):
        pp = aligned["PP"]
        crude = aligned["CRUDE"]
        pp_ret = returns["PP"]
        crude_ret = returns["CRUDE"]

        rolling_beta = rolling_beta_ols(pp_ret, crude_ret, window=12)
        if not rolling_beta.empty:
            fig = plot_rolling_beta(rolling_beta, title="Rolling 12M Beta: PP vs Crude")
            fig.savefig(PLOTS_DIR / "rolling_beta_pp_crude.png", bbox_inches="tight")
            plt.close(fig)

        fig = plot_beta_scaled_spread(
            pp,
            crude,
            beta=None,
            rolling_beta=rolling_beta["beta"] if "beta" in rolling_beta else None,
            title="Beta-Scaled Spread: PP vs Crude",
        )
        fig.savefig(PLOTS_DIR / "beta_scaled_spread_pp_crude.png", bbox_inches="tight")
        plt.close(fig)

        fig = plot_leadlag_heatmap(
            pp_ret,
            crude_ret,
            window=36,
            max_lag=12,
            title="Lead/Lag Correlation Heatmap: PP vs Crude",
        )
        fig.savefig(PLOTS_DIR / "leadlag_heatmap_pp_crude.png", bbox_inches="tight")
        plt.close(fig)

        fig = plot_scatter_returns(
            pp_ret,
            crude_ret,
            title="Monthly Log Returns: PP vs Crude",
            label_outliers=3,
        )
        fig.savefig(PLOTS_DIR / "scatter_returns_pp_crude.png", bbox_inches="tight")
        plt.close(fig)

        rolling = rolling_corr(pp_ret, crude_ret, window=12)
        corr_scan = cross_corr_scan(pp_ret, crude_ret, max_lag=12)
        rolling.to_csv(ARTIFACTS_DIR / "rolling_corr_pp_crude.csv", header=["rolling_corr"])
        corr_scan.to_csv(ARTIFACTS_DIR / "cross_corr_scan_pp_crude.csv", index=False)
    else:
        print("[WARN] Missing PP or Crude series; skipping joint diagnostics.")


if __name__ == "__main__":
    main()
