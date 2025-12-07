"""
Utility functions for polypropylene EDA workflows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

DATE_CANDIDATES = ["date", "month", "time", "timestamp"]
DATE_FORMATS = ["%b-%y", "%b-%Y", "%Y-%m", "%Y-%m-%d", "%d-%b-%Y"]
VALUE_CANDIDATES = ["price", "value", "index", "close", "avg", "average"]
OPTIONAL_COLUMNS = ["currency", "unit", "region", "source"]
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}


def _ensure_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def _pick_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        name = col.lower()
        for candidate in candidates:
            if candidate in name:
                return col
    return None


def _clean_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "iuf":
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def find_candidate_files(data_dir: str | Path, patterns: Sequence[str]) -> List[Path]:
    """
    Discover candidate data files under data_dir whose filenames contain any pattern.
    """
    root = _ensure_path(data_dir)
    if not root.exists():
        return []
    lowered_patterns = [p.lower() for p in patterns]
    matches: List[Path] = []
    for ext in ALLOWED_EXTENSIONS:
        for path in root.rglob(f"*{ext}"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not lowered_patterns or any(pattern in name for pattern in lowered_patterns):
                matches.append(path)
    return sorted(matches)


def load_series(path: Path, series_label: str) -> pd.DataFrame:
    """
    Load a single time-series file into a normalized DataFrame with metadata.
    """
    ext = path.suffix.lower()
    read_kwargs: Dict[str, object] = {}
    if ext == ".tsv":
        read_kwargs["sep"] = "\t"
    if ext in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path, **read_kwargs)

    original_rows = len(frame)
    date_column = _pick_column(frame.columns, DATE_CANDIDATES)
    if date_column is None:
        raise ValueError(f"No date column found in {path.name}")

    parsed_dates = pd.to_datetime(frame[date_column], errors="coerce")
    if parsed_dates.notna().sum() == 0:
        for fmt in DATE_FORMATS:
            parsed_dates = pd.to_datetime(frame[date_column], format=fmt, errors="coerce")
            if parsed_dates.notna().sum():
                break
    frame[date_column] = parsed_dates
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)
    frame = frame.set_index(date_column)

    value_column = _pick_column(frame.columns, VALUE_CANDIDATES)
    if value_column is None:
        numeric_candidates = [
            col for col in frame.columns if frame[col].dtype.kind in {"i", "u", "f"}
        ]
        value_column = numeric_candidates[0] if numeric_candidates else None
    if value_column is None:
        raise ValueError(f"Unable to infer value column for {path}")

    value_series = _clean_numeric(frame[value_column])
    frame = frame.assign(value=value_series).dropna(subset=["value"])

    optional_values: Dict[str, Optional[str]] = {}
    for optional in OPTIONAL_COLUMNS:
        match = _pick_column(frame.columns, [optional])
        if match:
            unique_values = frame[match].dropna().unique()
            if unique_values.size == 1:
                optional_values[optional] = str(unique_values[0])
            elif unique_values.size > 1:
                optional_values[optional] = f"various ({unique_values.size})"
            else:
                optional_values[optional] = None

    series = frame[["value"]].copy()
    series.attrs.update(
        {
            "series_name": series_label,
            "source_path": str(path),
            "rows_raw": original_rows,
            "rows_clean": len(series),
            "value_column": value_column,
            "date_column": date_column,
            "inferred_unit": optional_values.get("unit"),
            "inferred_currency": optional_values.get("currency"),
            "inferred_region": optional_values.get("region"),
            "start": series.index.min(),
            "end": series.index.max(),
        }
    )
    return series


def to_monthly(df: pd.DataFrame, how: str = "mean") -> pd.Series:
    """
    Resample a value series to monthly frequency (month start).
    """
    if "value" not in df.columns:
        raise ValueError("Expected column 'value' in DataFrame.")
    agg = df["value"].resample("MS")
    if how == "mean":
        monthly = agg.mean()
    elif how == "median":
        monthly = agg.median()
    elif how == "last":
        monthly = agg.last()
    else:
        raise ValueError(f"Unsupported aggregation: {how}")
    monthly.name = df.attrs.get("series_name", df.columns[0])
    monthly.attrs.update(df.attrs)
    return monthly


def normalize_base_100(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each series so the first non-null observation equals 100.
    """
    normalized = df.copy()
    for col in normalized.columns:
        series = normalized[col].dropna()
        if series.empty:
            normalized[col] = np.nan
            continue
        base = series.iloc[0]
        if pd.isna(base) or np.isclose(base, 0.0):
            normalized[col] = np.nan
            continue
        normalized[col] = normalized[col] / base * 100.0
    return normalized


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns for positive-valued series.
    """
    safe = df.where(df > 0)
    log_prices = np.log(safe)
    returns = log_prices.diff()
    return returns


def build_audit_table(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Build audit metadata for a dictionary of monthly series.
    """
    records: List[Dict[str, object]] = []
    for label, series in series_dict.items():
        data = series.dropna()
        first = data.index.min() if not data.empty else None
        last = data.index.max() if not data.empty else None
        count = len(data)
        expected_len = 0
        missing_pct = np.nan
        if first is not None and last is not None:
            expected_idx = pd.date_range(first, last, freq="MS")
            expected_len = len(expected_idx)
            if expected_len > 0:
                coverage = count / expected_len
                missing_pct = (1.0 - coverage) * 100
        attrs = series.attrs if hasattr(series, "attrs") else {}
        records.append(
            {
                "series": label,
                "first_date": first,
                "last_date": last,
                "count": count,
                "missing_pct": missing_pct,
                "frequency_note": f"monthly (resampled via mean)",
                "currency": attrs.get("inferred_currency"),
                "unit": attrs.get("inferred_unit"),
                "source_path": attrs.get("source_path"),
            }
        )
    return pd.DataFrame(records)


def rolling_corr(
    y: pd.Series, x: pd.Series, window: int = 12
) -> pd.Series:
    """
    Rolling correlation between y and x.
    """
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    return aligned["y"].rolling(window).corr(aligned["x"])


def cross_corr_scan(
    y: pd.Series, x: pd.Series, max_lag: int = 12
) -> pd.DataFrame:
    """
    Compute correlations across leads/lags.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_aligned = aligned.iloc[:, 0]
    x_aligned = aligned.iloc[:, 1]
    lags = range(-max_lag, max_lag + 1)
    rows = []
    for lag in lags:
        if lag > 0:
            corr = y_aligned.corr(x_aligned.shift(lag))
        elif lag < 0:
            corr = y_aligned.shift(-lag).corr(x_aligned)
        else:
            corr = y_aligned.corr(x_aligned)
        rows.append({"lag": lag, "corr": corr})
    return pd.DataFrame(rows)


def rolling_beta_ols(
    y: pd.Series, x: pd.Series, window: int = 12
) -> pd.DataFrame:
    """
    Rolling OLS beta of y on x.
    """
    combined = pd.concat([y, x], axis=1).dropna()
    if combined.empty:
        return pd.DataFrame(columns=["beta", "stderr", "lower", "upper", "r2", "n"])
    results = []
    end_indices = combined.index
    for end in range(window, len(combined) + 1):
        window_data = combined.iloc[end - window : end]
        if window_data.shape[0] < window:
            continue
        model = sm.OLS(window_data.iloc[:, 0], sm.add_constant(window_data.iloc[:, 1]))
        res = model.fit()
        conf = res.conf_int(alpha=0.05)
        beta_val = res.params.iloc[1]
        stderr = res.bse.iloc[1]
        lower = conf.iloc[1, 0]
        upper = conf.iloc[1, 1]
        results.append(
            {
                "index": window_data.index[-1],
                "beta": beta_val,
                "stderr": stderr,
                "lower": lower,
                "upper": upper,
                "r2": res.rsquared,
                "n": int(res.nobs),
            }
        )
    if not results:
        return pd.DataFrame(columns=["beta", "stderr", "lower", "upper", "r2", "n"])
    frame = pd.DataFrame(results).set_index("index")
    return frame


def _prepare_plot() -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    return fig


def plot_normalized_levels(df_norm: pd.DataFrame, title: str) -> plt.Figure:
    """
    Plot normalized level series. PGP is drawn dashed/transparent to avoid overlap with PP.
    """
    fig = _prepare_plot()
    ax = fig.axes[0]
    for column in df_norm.columns:
        style = {"linewidth": 2}
        if column.upper() == "PP":
            style.update({"linestyle": "-", "alpha": 1.0, "zorder": 3})
        elif column.upper() == "PGP":
            style.update({"linestyle": "--", "alpha": 0.7})
        ax.plot(df_norm.index, df_norm[column], label=column, **style)
    ax.set_title(title)
    ax.set_ylabel("Index (Base = 100)")
    ax.set_xlabel("Date")
    ax.legend()
    return fig


def _estimate_beta_from_returns(pp: pd.Series, crude: pd.Series) -> float:
    aligned = pd.concat([pp, crude], axis=1).dropna()
    if aligned.empty:
        return np.nan
    positive = aligned[(aligned > 0).all(axis=1)]
    log_returns = np.log(positive).diff().dropna()
    if log_returns.empty:
        return np.nan
    model = sm.OLS(log_returns.iloc[:, 0], sm.add_constant(log_returns.iloc[:, 1]))
    res = model.fit()
    return float(res.params.iloc[1])


def plot_beta_scaled_spread(
    pp: pd.Series,
    crude: pd.Series,
    beta: Optional[float],
    rolling_beta: Optional[pd.Series],
    title: str,
) -> plt.Figure:
    """
    Plot beta-scaled spread between PP and crude.
    """
    combined = pd.concat([pp.rename("PP"), crude.rename("Crude")], axis=1).dropna()
    df_norm = normalize_base_100(combined)
    pp_norm = df_norm["PP"]
    crude_norm = df_norm["Crude"]
    if beta is None or np.isnan(beta):
        beta = _estimate_beta_from_returns(pp_norm, crude_norm)
    spread = pp_norm - beta * crude_norm

    fig = _prepare_plot()
    ax = fig.axes[0]
    ax.plot(pp_norm.index, pp_norm, label="PP (norm)", linewidth=2)
    ax.plot(crude_norm.index, crude_norm * beta, label=f"Crude × β ({beta:.2f})", linewidth=2)
    ax.plot(spread.index, spread, label="Beta-scaled spread", linewidth=2, linestyle="--")

    if rolling_beta is not None and not rolling_beta.dropna().empty:
        rb = rolling_beta.reindex(spread.index).ffill()
        spread_rb = pp_norm - rb * crude_norm
        ax.plot(
            spread_rb.index,
            spread_rb,
            label="Rolling β spread",
            linewidth=1.5,
            linestyle=":",
            color="tab:red",
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Index / Spread")
    ax.set_xlabel("Date")
    ax.legend()
    return fig


def plot_rolling_beta(rolling: pd.DataFrame, title: str) -> plt.Figure:
    """
    Plot rolling beta with 95% confidence interval and R².
    """
    fig = _prepare_plot()
    ax = fig.axes[0]
    ax.plot(rolling.index, rolling["beta"], label="β", color="tab:blue", linewidth=2)
    ax.fill_between(
        rolling.index,
        rolling["lower"],
        rolling["upper"],
        color="tab:blue",
        alpha=0.2,
        label="95% CI",
    )
    ax.set_title(title)
    ax.set_ylabel("β")
    ax.set_xlabel("Date")

    if "r2" in rolling.columns:
        ax2 = ax.twinx()
        ax2.plot(
            rolling.index,
            rolling["r2"],
            color="tab:orange",
            linewidth=1.5,
            label="R²",
        )
        ax2.set_ylabel("R²")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend()
    return fig


def plot_leadlag_heatmap(
    y_ret: pd.Series,
    x_ret: pd.Series,
    window: int = 36,
    max_lag: int = 12,
    title: str = "",
) -> plt.Figure:
    """
    Plot a rolling lead/lag correlation heatmap.
    """
    combined = pd.concat([y_ret.rename("y"), x_ret.rename("x")], axis=1).dropna()
    if combined.empty:
        raise ValueError("Insufficient data for lead/lag heatmap.")
    lags = list(range(-max_lag, max_lag + 1))
    records = []
    for end in range(window, len(combined) + 1):
        window_data = combined.iloc[end - window : end]
        row = {"date": window_data.index[-1]}
        for lag in lags:
            if lag > 0:
                corr = window_data["y"].corr(window_data["x"].shift(lag))
            elif lag < 0:
                corr = window_data["y"].shift(-lag).corr(window_data["x"])
            else:
                corr = window_data["y"].corr(window_data["x"])
            row[lag] = corr
        records.append(row)
    heatmap_df = pd.DataFrame(records).set_index("date")

    fig = _prepare_plot()
    ax = fig.axes[0]
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.axvline(max_lag + 0.5, color="black", linewidth=1)
    ax.set_xticklabels([int(tick.get_text()) for tick in ax.get_xticklabels()])
    ax.set_title(title)
    ax.set_xlabel("Lag (months, driver leads if >0)")
    ax.set_ylabel("Window end")
    return fig


def plot_scatter_returns(
    y_ret: pd.Series,
    x_ret: pd.Series,
    title: str,
    label_outliers: int = 3,
) -> plt.Figure:
    """
    Scatter plot of returns with OLS fit and volatility shading.
    """
    combined = pd.concat([y_ret.rename("y"), x_ret.rename("x")], axis=1).dropna()
    if combined.empty:
        raise ValueError("No overlapping returns for scatter plot.")

    rolling_std = combined["x"].rolling(12).std()
    bins = pd.qcut(rolling_std.dropna(), q=3, labels=["Low", "Medium", "High"])
    volatility_label = bins.reindex(combined.index, method="nearest")
    color_map = {"Low": "tab:green", "Medium": "tab:orange", "High": "tab:red"}
    colors = [color_map.get(volatility_label.get(idx), "tab:blue") for idx in combined.index]

    model = sm.OLS(combined["y"], sm.add_constant(combined["x"]))
    res = model.fit()
    fit_line = res.params["const"] + res.params["x"] * combined["x"]
    residuals = (combined["y"] - fit_line).abs()
    outlier_idx = residuals.nlargest(label_outliers).index

    fig = _prepare_plot()
    ax = fig.axes[0]
    ax.scatter(combined["x"], combined["y"], c=colors, alpha=0.7, edgecolor="none")
    ax.plot(combined["x"], fit_line, color="black", linewidth=2, label="OLS fit")
    for idx in outlier_idx:
        ax.annotate(idx.strftime("%Y-%m"), (combined.loc[idx, "x"], combined.loc[idx, "y"]))
    ax.set_title(title)
    ax.set_xlabel(x_ret.name or "Driver returns")
    ax.set_ylabel(y_ret.name or "Series returns")
    ax.legend()
    return fig


def plot_calendar_heatmap(ret: pd.Series, title: str) -> plt.Figure:
    """
    Calendar-style heatmap of returns.
    """
    data = ret.dropna()
    if data.empty:
        raise ValueError("No data for calendar heatmap.")
    df = data.to_frame(name="ret")
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    pivot = df.pivot_table(index="Year", columns="Month", values="ret", aggfunc="mean")
    fig = _prepare_plot()
    ax = fig.axes[0]
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Return"})
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return fig


def plot_monthly_box(ret: pd.Series, title: str) -> plt.Figure:
    """
    Boxplot of returns grouped by calendar month.
    """
    df = ret.dropna().to_frame("ret")
    df["month"] = df.index.month
    fig = _prepare_plot()
    ax = fig.axes[0]
    sns.boxplot(data=df, x="month", y="ret", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Return")
    return fig


def plot_rolling_vol(
    ret_dict: Dict[str, pd.Series], window: int = 12, title: str = ""
) -> plt.Figure:
    """
    Plot rolling volatility for multiple return series.
    """
    fig = _prepare_plot()
    ax = fig.axes[0]
    for label, series in ret_dict.items():
        ax.plot(series.index, series.rolling(window).std(), label=label, linewidth=2)
    ax.set_title(title or f"Rolling {window}-month volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std dev")
    ax.legend()
    return fig


def plot_drawdown(level: pd.Series, title: str) -> plt.Figure:
    """
    Plot drawdown curve for a level series.
    """
    data = level.dropna()
    cummax = data.cummax()
    drawdown = (data / cummax) - 1.0
    fig = _prepare_plot()
    ax = fig.axes[0]
    ax.plot(drawdown.index, drawdown, label="Drawdown", color="tab:red")
    ax.fill_between(drawdown.index, drawdown, color="tab:red", alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend()
    return fig


def plot_yoy(level_dict: Dict[str, pd.Series], title: str) -> plt.Figure:
    """
    Plot year-over-year percentage change for level series.
    """
    fig = _prepare_plot()
    ax = fig.axes[0]
    for label, series in level_dict.items():
        yoy = series.pct_change(periods=12) * 100
        ax.plot(yoy.index, yoy, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel("YoY % change")
    ax.set_xlabel("Date")
    ax.legend()
    return fig
