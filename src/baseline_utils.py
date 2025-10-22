"""
Baseline forecasting utilities for polypropylene analysis.
"""
from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL


def train_test_split_ts(
    series: pd.Series, test_size_months: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Deterministic train/test split for time series.
    """
    series = series.dropna()
    if series.empty:
        raise ValueError("Series is empty; cannot split.")
    split_point = max(len(series) - test_size_months, int(len(series) * 0.8))
    train = series.iloc[:split_point]
    test = series.iloc[split_point:]
    if test.empty:
        raise ValueError("Test set ended up empty; reduce test_size_months.")
    return train, test


def rolling_origin_backtest(
    y: pd.Series,
    X: pd.DataFrame,
    horizon: int = 1,
    start_min_obs: int = 36,
) -> pd.DataFrame:
    """
    Rolling-origin expanding-window forecast evaluation.
    """
    if horizon != 1:
        raise NotImplementedError("Only horizon=1 supported currently.")
    y = y.dropna()
    X = X.loc[y.index].dropna()
    records = []
    exog_names = None
    for i in range(start_min_obs, len(y)):
        train_y = y.iloc[:i]
        train_X = X.iloc[:i]
        test_X = X.iloc[i : i + horizon]
        if len(train_y) < start_min_obs or test_X.empty:
            continue
        train_exog = sm.add_constant(train_X, has_constant="add")
        model = sm.OLS(train_y, train_exog).fit()
        if exog_names is None:
            exog_names = model.model.exog_names
        test_exog = sm.add_constant(test_X, has_constant="add")
        if exog_names is not None:
            test_exog = test_exog.reindex(columns=exog_names, fill_value=0)
        pred = model.predict(test_exog)
        date = pred.index[-1]
        records.append(
            {
                "date": date,
                "y_true": y.loc[date],
                "y_pred": pred.iloc[-1],
                "model": "OLS_lagged_crude",
            }
        )
    return pd.DataFrame(records)


def forecast_naive(levels: pd.Series) -> pd.Series:
    """
    Naive forecast: y_hat_t = y_{t-1}.
    """
    return levels.shift(1)


def forecast_seasonal_naive(levels: pd.Series, season: int = 12) -> pd.Series:
    """
    Seasonal naive forecast: y_hat_t = y_{t-season}.
    """
    return levels.shift(season)


def forecast_holt_winters(levels: pd.Series) -> pd.Series:
    """
    Additive ETS (Holt-Winters) forecast with seasonal period.
    """
    if len(levels.dropna()) < 24:
        raise ValueError("Not enough data for Holt-Winters (need >=24 observations).")
    seasonal_periods = 12 if len(levels) >= 36 else None
    model = ExponentialSmoothing(
        levels,
        trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
    )
    fitted = model.fit(optimized=True)
    return fitted.fittedvalues


def fit_ols_returns(
    y_ret: pd.Series, x_ret: pd.Series, lags: int = 3
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """
    Fit OLS of y_ret on current and lagged x_ret values.
    """
    data = pd.concat([y_ret.rename("y"), x_ret.rename("x0")], axis=1)
    for lag in range(1, lags + 1):
        data[f"x{lag}"] = x_ret.shift(lag)
    data = data.dropna()
    y = data["y"]
    X = data.drop(columns="y")
    exog = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, exog).fit()
    return model, X


def metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE.
    """
    aligned = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1).dropna()
    abs_err = (aligned["y"] - aligned["yhat"]).abs()
    mae = abs_err.mean()
    rmse = np.sqrt(((aligned["y"] - aligned["yhat"]) ** 2).mean())
    mape = (abs_err / aligned["y"].replace(0, np.nan)).dropna().mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _prepare_fig() -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, _ = plt.subplots(figsize=(10, 5), dpi=150)
    return fig


def plot_forecast_vs_actual(
    dates: pd.Index,
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
) -> plt.Figure:
    """
    Plot actual vs forecasted series.
    """
    fig = _prepare_fig()
    ax = fig.axes[0]
    ax.plot(dates, y_true.reindex(dates), label="Actual", linewidth=2)
    ax.plot(dates, y_pred.reindex(dates), label="Forecast", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("PP Level")
    ax.legend()
    return fig


def plot_residuals(resid: pd.Series, title: str) -> plt.Figure:
    """
    Plot residual series over time.
    """
    fig = _prepare_fig()
    ax = fig.axes[0]
    ax.plot(resid.index, resid, label="Residual", color="tab:red")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.legend()
    return fig


def plot_stl(levels: pd.Series, title: str) -> plt.Figure:
    """
    STL decomposition plot (trend, seasonal, residual).
    """
    if len(levels.dropna()) < 24:
        raise ValueError("Insufficient data for STL decomposition.")
    stl = STL(levels.dropna(), period=12)
    result = stl.fit()
    fig = result.plot()
    fig.set_size_inches(10, 6)
    fig.set_dpi(150)
    fig.suptitle(title)
    return fig
