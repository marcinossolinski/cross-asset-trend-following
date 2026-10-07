"""Daily backtest engine with month-end rebalancing, position drift and costs.

Timing convention (the most common source of look-ahead bias):
    * signals, volatilities and covariances are computed with data up to the
      close of rebalance date d;
    * the new positions are traded at that close, paying transaction costs on
      the change versus the *drifted* positions held before the trade;
    * the first return they earn is the return of the next trading day d+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ASSET_CLASS, COST_BPS, StrategyConfig
from .portfolio import cost_rates, target_weights
from .risk import ewma_cov_at, ewma_vol
from .signals import rebalance_dates, tsmom_signal


@dataclass
class BacktestResult:
    config: StrategyConfig
    returns: pd.Series            # daily net excess return of the strategy
    gross_returns: pd.Series      # before transaction costs
    costs: pd.Series              # cost paid each day (as a return)
    target_weights: pd.DataFrame  # weights set at each rebalance
    turnover: pd.Series           # sum |trade| at each rebalance
    contributions: pd.DataFrame   # daily P&L contribution by asset

    @property
    def monthly(self) -> pd.Series:
        return (1 + self.returns).resample("ME").prod() - 1

    def class_contributions(self) -> pd.DataFrame:
        return self.contributions.T.groupby(ASSET_CLASS).sum().T


class Precomputed:
    """Data that does not depend on the signal parameters, cached across a parameter grid."""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.dates = rebalance_dates(returns.index)
        self.history = returns.notna().cumsum()
        # share of days without a price change over the last quarter (pegs, stale data)
        self.stale = (returns == 0).astype(float).where(returns.notna()).rolling(63, min_periods=63).mean()
        self._vol: dict = {}
        self._cov: dict = {}

    def vol(self, com: int) -> pd.DataFrame:
        if com not in self._vol:
            self._vol[com] = ewma_vol(self.returns, com)
        return self._vol[com]

    def cov(self, halflife: int) -> dict:
        if halflife not in self._cov:
            self._cov[halflife] = ewma_cov_at(self.returns, self.dates, halflife)
        return self._cov[halflife]


def run_backtest(returns: pd.DataFrame, cfg: StrategyConfig,
                 pre: Precomputed | None = None) -> BacktestResult:
    pre = pre or Precomputed(returns)
    dates = pre.dates[pre.dates >= pd.Timestamp(cfg.trading_start)]
    vol = pre.vol(cfg.vol_com_days)
    covs = pre.cov(cfg.cov_halflife_days)
    # signals use the full month-end grid so that look-backs before the first trade are available
    signals = tsmom_signal(returns, pre.dates, cfg.lookbacks, cfg.signal, ann_vol=vol).loc[dates]

    targets = pd.DataFrame(
        [target_weights(signals.loc[d], vol.loc[d], covs[d], pre.history.loc[d], pre.stale.loc[d], cfg)
         for d in dates],
        index=dates,
    )

    rates = pd.Series(cost_rates(returns.columns, COST_BPS)) * cfg.cost_multiplier
    r = returns.fillna(0.0).to_numpy()
    idx = returns.index
    rebal_pos = {idx.get_loc(d): d for d in dates}
    rate_arr = rates.reindex(returns.columns).to_numpy()

    n_days, n_assets = r.shape
    w = np.zeros(n_assets)                     # current (drifted) weights
    gross = np.zeros(n_days)
    cost = np.zeros(n_days)
    contrib = np.zeros((n_days, n_assets))
    turnover = {}

    for t in range(n_days):
        # 1) P&L of positions held from yesterday's close to today's close
        pnl = w * r[t]
        contrib[t] = pnl
        gross[t] = pnl.sum()
        # 2) let positions drift with prices (futures-style notional exposure)
        w = w * (1.0 + r[t]) / (1.0 + gross[t])
        # 3) rebalance at today's close if it is a month end
        if t in rebal_pos:
            new_w = targets.loc[rebal_pos[t]].to_numpy()
            trade = np.abs(new_w - w)
            cost[t] = float(trade @ rate_arr)
            turnover[rebal_pos[t]] = float(trade.sum())
            w = new_w

    # the track record starts the day after the first trade; the cost of that
    # initial trade is charged to the first day so that no cost is dropped
    first = dates[targets.abs().sum(axis=1) > 0][0]
    first_pos = idx.get_loc(first)
    cost[first_pos + 1] += cost[first_pos]
    cost[: first_pos + 1] = 0.0
    keep = idx > first
    net = pd.Series(gross - cost, index=idx)[keep]
    return BacktestResult(
        config=cfg,
        returns=net,
        gross_returns=pd.Series(gross, index=idx)[keep],
        costs=pd.Series(cost, index=idx)[keep],
        target_weights=targets,
        turnover=pd.Series(turnover),
        contributions=pd.DataFrame(contrib, index=idx, columns=returns.columns)[keep],
    )
