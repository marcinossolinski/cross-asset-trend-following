"""Research analyses: robustness, out-of-sample selection, crisis behaviour, data biases."""

from __future__ import annotations

from itertools import product
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Precomputed, run_backtest
from .config import StrategyConfig
from .metrics import sharpe, to_monthly


# --------------------------------------------------------------------------- #
# Parameter grid (also defines the number of trials for the Deflated Sharpe)
# --------------------------------------------------------------------------- #
def parameter_grid(base: StrategyConfig,
                   lookbacks: Sequence[int] = (1, 3, 6, 9, 12, 18, 24),
                   vol_coms: Sequence[int] = (20, 60, 120)) -> list[StrategyConfig]:
    """Every configuration a researcher could plausibly have tried."""
    grid = [base.with_(lookbacks=(lb,), vol_com_days=c) for lb, c in product(lookbacks, vol_coms)]
    grid += [base.with_(vol_com_days=c) for c in vol_coms]                      # blended signal
    grid += [base.with_(lookbacks=(lb,), signal="scaled") for lb in lookbacks]  # continuous signal
    grid += [base.with_(signal="scaled")]
    return grid


def run_grid(returns: pd.DataFrame, configs: Sequence[StrategyConfig],
             pre: Precomputed) -> Dict[str, BacktestResult]:
    return {c.label: run_backtest(returns, c, pre) for c in configs}


def sensitivity_table(results: Dict[str, BacktestResult], lookbacks: Sequence[int],
                      vol_coms: Sequence[int]) -> pd.DataFrame:
    """Sharpe ratio for each single look-back x volatility window (sign signal)."""
    tab = pd.DataFrame(index=list(lookbacks), columns=list(vol_coms), dtype=float)
    for lb, c in product(lookbacks, vol_coms):
        tab.loc[lb, c] = sharpe(results[f"TSMOM[{lb}m|sign|com{c}]"].monthly)
    tab.index.name = "lookback_months"
    tab.columns.name = "vol_com_days"
    return tab


# --------------------------------------------------------------------------- #
# Walk-forward selection
# --------------------------------------------------------------------------- #
def walk_forward(results: Dict[str, BacktestResult], first_year: int = 1985,
                 min_train_years: int = 10) -> tuple[pd.Series, pd.DataFrame]:
    """Each January pick the configuration with the best Sharpe on all prior data,
    then hold it for the calendar year. Returns stitched out-of-sample monthly
    returns and the selection log.

    Every candidate's return at month t uses only information up to t-1, so
    choosing among them with data before the year starts is genuinely out of sample.
    """
    monthly = pd.DataFrame({k: v.monthly for k, v in results.items()})
    last_year = monthly.index.max().year
    oos, log = [], []
    for year in range(first_year, last_year + 1):
        train = monthly[monthly.index.year < year]
        if train.index.year.nunique() < min_train_years:
            continue
        srs = train.apply(sharpe)
        best = srs.idxmax()
        test = monthly.loc[monthly.index.year == year, best]
        oos.append(test)
        log.append({"year": year, "selected": best, "in_sample_sharpe": srs[best],
                    "oos_return": float((1 + test).prod() - 1)})
    return pd.concat(oos), pd.DataFrame(log).set_index("year")


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def cost_sensitivity(returns: pd.DataFrame, cfg: StrategyConfig, pre: Precomputed,
                     multipliers: Sequence[float] = (0, 1, 2, 5, 10)) -> pd.DataFrame:
    rows = []
    for m in multipliers:
        res = run_backtest(returns, cfg.with_(cost_multiplier=m), pre)
        years = len(res.monthly) / 12
        rows.append({"cost_multiplier": m, "sharpe": sharpe(res.monthly),
                     "ann_cost_drag": res.costs.sum() / years})
    return pd.DataFrame(rows).set_index("cost_multiplier")


# --------------------------------------------------------------------------- #
# Crisis behaviour
# --------------------------------------------------------------------------- #
def equity_drawdown_episodes(equity_tr_daily: pd.Series, strategy_daily: pd.Series,
                             threshold: float = 0.15) -> pd.DataFrame:
    """Every peak-to-trough fall of US equities larger than ``threshold``.

    Episodes are detected mechanically from the data (no hand-picked crises).
    """
    common = equity_tr_daily.index.intersection(strategy_daily.index)
    eq = equity_tr_daily.loc[common].fillna(0.0)
    strat = strategy_daily.loc[common].fillna(0.0)
    wealth = (1 + eq).cumprod()
    dd = wealth / wealth.cummax() - 1

    # every new high starts a group; a group is "peak day + the drawdown that follows"
    group = (dd == 0).cumsum()
    episodes = []
    for _, seg in dd.groupby(group):
        if seg.min() <= -threshold:
            episodes.append((seg.index[0], seg.idxmin()))

    rows = []
    for peak_d, trough_d in episodes:
        window = (eq.index > peak_d) & (eq.index <= trough_d)
        rows.append({
            "peak": peak_d.date(), "trough": trough_d.date(),
            "us_equities": float((1 + eq[window]).prod() - 1),
            "trend_strategy": float((1 + strat[window]).prod() - 1),
        })
    return pd.DataFrame(rows)


def smile(strategy_daily: pd.Series, equity_excess_daily: pd.Series) -> pd.DataFrame:
    """Non-overlapping quarterly returns, for the 'trend-following smile' plot."""
    q = lambda s: (1 + s).resample("QE").prod() - 1  # noqa: E731
    df = pd.concat({"equity": q(equity_excess_daily), "strategy": q(strategy_daily)}, axis=1).dropna()
    return df[df.index >= strategy_daily.index.min()]


def subperiod_sharpe(monthly: pd.Series, freq_years: int = 10) -> pd.Series:
    decade = (monthly.index.year // freq_years) * freq_years
    return monthly.groupby(decade).apply(sharpe).rename("sharpe")


# --------------------------------------------------------------------------- #
# Data-integrity experiment: averaged prices create spurious momentum
# --------------------------------------------------------------------------- #
def averaging_bias(returns: pd.DataFrame) -> pd.DataFrame:
    """Compare a 1-month trend signal on month-end prices vs monthly-average prices.

    Many free data sets (gold, Treasury yields, commodity indices) only publish
    monthly averages. The 'return' between two monthly averages overlaps with
    the second half of the month in which the signal is formed, so it is partly
    known in advance. This function measures how much that inflates results.
    """
    rows = []
    for c in returns.columns:
        px = (1 + returns[c].dropna()).cumprod()
        versions = {
            "month_end": px.resample("ME").last(),
            "monthly_average": px.resample("ME").mean(),
        }
        for name, p in versions.items():
            r = p.pct_change().dropna()
            vol = r.rolling(36, min_periods=24).std().shift(1)
            pos = np.sign(r.shift(1)) / vol
            strat = (pos * r).dropna()
            rows.append({"asset": c, "price_series": name,
                         "lag1_autocorr": r.autocorr(1),
                         "trend_1m_sharpe": sharpe(strat)})
    return pd.DataFrame(rows)


def monthly_from(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({k: to_monthly(v.returns) for k, v in results.items()})
