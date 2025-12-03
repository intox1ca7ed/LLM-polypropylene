"""
Utilities for sentiment-augmented PP forecasting.

Responsibilities:
- Define baseline vs sentiment model specs.
- Build design matrices with lags.
- Run rolling-origin backtests with expanding windows.
- Compute forecast metrics.
- Fit full-sample models for coefficient inspection.
- Provide plotting helpers for forecasts, errors, and coefficients.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

MIN_REQUIRED_ROWS = 24  # fallback guard for short series


@dataclass
class ModelSpec:
    name: str
    target: str
    regressors: List[str]
    target_lags: List[int]
    reg_lags: Dict[str, List[int]]


def _available_cols(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def define_model_specs(df: pd.DataFrame) -> List[ModelSpec]:
    """
    Build a list of baseline and sentiment model specs, adapting to available columns.
    """
    target = "resid_PP" if "resid_PP" in df.columns else "ret_PP"
    crude_ret = "ret_CRUDE" if "ret_CRUDE" in df.columns else None
    pgp_ret = "ret_PGP" if "ret_PGP" in df.columns else None

    base_regs = [c for c in [crude_ret, pgp_ret] if c]
    sentiment_hybrid = "index_hybrid" if "index_hybrid" in df.columns else None
    section_regs = _available_cols(df, ["index_demand", "index_supply", "index_price_outlook"])

    specs: List[ModelSpec] = []

    specs.append(
        ModelSpec(
            name="A_baseline",
            target=target,
            regressors=base_regs,
            target_lags=[1],
            reg_lags={r: [1] for r in base_regs},
        )
    )

    if sentiment_hybrid:
        regs_b = base_regs + [sentiment_hybrid]
        reg_lags_b = {r: [1] for r in base_regs}
        reg_lags_b[sentiment_hybrid] = [0, 1]
        specs.append(
            ModelSpec(
                name="B_sentiment_hybrid",
                target=target,
                regressors=regs_b,
                target_lags=[1],
                reg_lags=reg_lags_b,
            )
        )

    if sentiment_hybrid and section_regs:
        regs_c = base_regs + [sentiment_hybrid] + section_regs
        reg_lags_c = {r: [1] for r in base_regs}
        reg_lags_c[sentiment_hybrid] = [0, 1]
        for s in section_regs:
            reg_lags_c[s] = [0, 1]
        specs.append(
            ModelSpec(
                name="C_sentiment_sections",
                target=target,
                regressors=regs_c,
                target_lags=[1],
                reg_lags=reg_lags_c,
            )
        )

    usable_specs = [s for s in specs if s.regressors or s.target_lags]
    return usable_specs


def build_design_matrices(df: pd.DataFrame, spec: ModelSpec, max_lag: int = 2) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Apply lags to target and regressors and return aligned y, X.
    """
    if spec.target not in df.columns:
        raise ValueError(f"Target {spec.target} not in dataframe.")

    work = df.copy()

    y = work[spec.target].copy()
    lagged_parts = {}

    target_lags = [l for l in spec.target_lags if l > 0 and l <= max_lag]
    for lag in target_lags:
        lagged_parts[f"{spec.target}_lag{lag}"] = y.shift(lag)

    for reg in spec.regressors:
        lags = spec.reg_lags.get(reg, [0])
        for lag in lags:
            if lag > max_lag:
                continue
            col_name = f"{reg}_lag{lag}" if lag > 0 else reg
            lagged_parts[col_name] = work[reg].shift(lag) if lag > 0 else work[reg]

    X = pd.DataFrame(lagged_parts)
    combined = pd.concat([y, X], axis=1).dropna()
    y_aligned = combined[spec.target]
    X_aligned = combined.drop(columns=[spec.target])
    return y_aligned, X_aligned


def train_test_split_rolling(y: pd.Series, X: pd.DataFrame, test_size_months: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train/test indices for a simple last-N-months split.
    """
    n = len(y)
    test_size = min(test_size_months, max(n // 3, 1))
    split = n - test_size
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)
    return train_idx, test_idx


def rolling_origin_backtest(y: pd.Series, X: pd.DataFrame, model_name: str, test_size_months: int = 24, min_train_months: int = 36) -> pd.DataFrame:
    """
    Expanding-window rolling-origin backtest with 1-step-ahead forecasts.
    """
    if len(y) < max(test_size_months, MIN_REQUIRED_ROWS):
        warnings.warn(f"Series too short ({len(y)} rows); adjusting test_size to {max(6, len(y)//3)}.")
        test_size_months = max(6, len(y) // 3)
        min_train_months = max(12, len(y) // 4)

    train_idx, test_idx = train_test_split_rolling(y, X, test_size_months)
    results = []

    for idx in test_idx:
        train_mask = np.arange(0, idx)
        if len(train_mask) < min_train_months:
            continue
        model = LinearRegression()
        model.fit(X.iloc[train_mask], y.iloc[train_mask])
        pred = model.predict(X.iloc[[idx]])[0]
        results.append(
            {
                "date": y.index[idx],
                "y_true": y.iloc[idx],
                "y_pred": pred,
                "model_name": model_name,
                "train_end": y.index[idx - 1],
            }
        )

    return pd.DataFrame(results)


def run_models_backtest(df: pd.DataFrame, model_specs: List[ModelSpec], test_size_months: int = 24, min_train_months: int = 36) -> pd.DataFrame:
    """
    Run rolling backtests for all model specs and concatenate results.
    """
    all_results = []
    for spec in model_specs:
        try:
            y, X = build_design_matrices(df, spec, max_lag=max(max(spec.target_lags or [0] + sum(spec.reg_lags.values(), [])), 1))
            res = rolling_origin_backtest(y, X, spec.name, test_size_months=test_size_months, min_train_months=min_train_months)
            all_results.append(res)
        except Exception as exc:
            warnings.warn(f"Skipping {spec.name} due to error: {exc}")
    if not all_results:
        return pd.DataFrame(columns=["date", "y_true", "y_pred", "model_name"])
    return pd.concat(all_results, ignore_index=True)


def compute_forecast_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MAPE, and directional accuracy per model.
    """
    if backtest_df.empty:
        return pd.DataFrame()
    metrics = []
    for name, g in backtest_df.groupby("model_name"):
        errors = g["y_true"] - g["y_pred"]
        mae = errors.abs().mean()
        rmse = np.sqrt((errors**2).mean())
        mape = (errors.abs() / g["y_true"].replace(0, np.nan)).mean()
        directional = (np.sign(g["y_true"]) == np.sign(g["y_pred"])).mean()
        metrics.append(
            {
                "model_name": name,
                "N": len(g),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Directional_Accuracy": directional,
            }
        )
    return pd.DataFrame(metrics).sort_values("RMSE")


def extract_coefficients(df: pd.DataFrame, model_specs: List[ModelSpec]) -> pd.DataFrame:
    """
    Fit each model on the full sample and return coefficients.
    """
    rows = []
    for spec in model_specs:
        try:
            y, X = build_design_matrices(df, spec, max_lag=max(max(spec.target_lags or [0] + sum(spec.reg_lags.values(), [])), 1))
            model = LinearRegression()
            model.fit(X, y)
            for name, coef in zip(X.columns, model.coef_):
                rows.append({"model_name": spec.name, "variable": name, "coefficient": coef})
            rows.append({"model_name": spec.name, "variable": "intercept", "coefficient": model.intercept_})
        except Exception as exc:
            warnings.warn(f"Skipping coefficient extraction for {spec.name} due to error: {exc}")
    return pd.DataFrame(rows)


# ---- Plotting helpers ----

def plot_forecast_vs_actual(backtest_df: pd.DataFrame, model_names_to_show: List[str] | None = None) -> plt.Figure:
    if backtest_df.empty:
        raise ValueError("Backtest results are empty.")
    df = backtest_df.copy()
    if model_names_to_show:
        df = df[df["model_name"].isin(model_names_to_show)]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(df["date"], df["y_true"], label="Actual", color="black")
    for name, g in df.groupby("model_name"):
        ax.plot(g["date"], g["y_pred"], label=name, alpha=0.8)
    ax.set_title("Forecast vs Actual (test period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_forecast_errors(backtest_df: pd.DataFrame) -> plt.Figure:
    if backtest_df.empty:
        raise ValueError("Backtest results are empty.")
    df = backtest_df.copy()
    df["error"] = df["y_true"] - df["y_pred"]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    for name, g in df.groupby("model_name"):
        ax.plot(g["date"], g["error"], label=name, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Forecast errors over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (y_true - y_pred)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def plot_coefficients(coefs_df: pd.DataFrame) -> plt.Figure:
    if coefs_df.empty:
        raise ValueError("Coefficient table is empty.")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    pivot = coefs_df[coefs_df["variable"] != "intercept"].pivot(index="variable", columns="model_name", values="coefficient")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Model coefficients (full sample)")
    ax.set_ylabel("Coefficient")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_sentiment_vs_residual_zoom(df: pd.DataFrame, dates: List[pd.Timestamp]) -> List[plt.Figure]:
    """
    Plot residuals and index_hybrid around specific dates (event-style).
    """
    figs = []
    if df.empty or "index_hybrid" not in df.columns:
        return figs
    target_col = "resid_PP" if "resid_PP" in df.columns else "ret_PP"
    for date in dates:
        window = df.loc[(df.index >= date - pd.DateOffset(months=3)) & (df.index <= date + pd.DateOffset(months=3)), [target_col, "index_hybrid"]]
        if window.empty:
            continue
        sns.set_style("whitegrid")
        fig, ax1 = plt.subplots(figsize=(8, 4), dpi=120)
        ax1.plot(window.index, window[target_col], label=target_col, color="steelblue")
        ax1.set_ylabel(target_col)
        ax2 = ax1.twinx()
        ax2.plot(window.index, window["index_hybrid"], label="index_hybrid", color="darkred")
        ax2.set_ylabel("index_hybrid")
        ax1.set_title(f"{target_col} and sentiment around {date.date()}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.autofmt_xdate()
        figs.append(fig)
    return figs
