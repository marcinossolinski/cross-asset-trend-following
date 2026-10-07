"""Performance and risk statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_monthly(daily: pd.Series) -> pd.Series:
    return (1.0 + daily).resample("ME").prod() - 1.0


def drawdown(daily: pd.Series) -> pd.Series:
    wealth = (1.0 + daily.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def sharpe(monthly: pd.Series) -> float:
    m = monthly.dropna()
    return float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if m.std() > 0 else np.nan


def sortino(monthly: pd.Series) -> float:
    m = monthly.dropna()
    downside = np.sqrt((np.minimum(m, 0.0) ** 2).mean()) * np.sqrt(12)
    return float(m.mean() * 12 / downside) if downside > 0 else np.nan


def cagr(monthly: pd.Series) -> float:
    m = monthly.dropna()
    return float((1 + m).prod() ** (12 / len(m)) - 1)


def summary(daily: pd.Series, turnover: pd.Series | None = None,
            costs: pd.Series | None = None) -> dict:
    """Standard set of statistics, computed on monthly returns unless stated."""
    m = to_monthly(daily).dropna()
    dd = drawdown(daily)
    years = len(m) / 12
    out = {
        "start": m.index.min().strftime("%Y-%m"),
        "end": m.index.max().strftime("%Y-%m"),
        "ann_return": m.mean() * 12,
        "cagr": cagr(m),
        "ann_vol": m.std(ddof=1) * np.sqrt(12),
        "sharpe": sharpe(m),
        "sortino": sortino(m),
        "max_drawdown": dd.min(),
        "calmar": cagr(m) / abs(dd.min()) if dd.min() < 0 else np.nan,
        "skew": m.skew(),
        "excess_kurtosis": m.kurt(),
        "hit_rate": (m > 0).mean(),
        "best_month": m.max(),
        "worst_month": m.min(),
    }
    if turnover is not None:
        out["annual_turnover"] = turnover.sum() / years
    if costs is not None:
        out["annual_cost_drag"] = costs.sum() / years
    return out
