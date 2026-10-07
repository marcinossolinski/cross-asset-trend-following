"""Portfolio construction: risk budgets, ex-ante volatility targeting and limits."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .config import ASSET_CLASS, StrategyConfig


def target_weights(signal: pd.Series, asset_vol: pd.Series, cov: np.ndarray,
                   history_days: pd.Series, stale_share: pd.Series, cfg: StrategyConfig,
                   asset_class: Mapping[str, str] = ASSET_CLASS) -> pd.Series:
    """Weights (as a fraction of capital, i.e. notional exposure) for one rebalance date.

    1. Tradability: enough history, a valid signal, volatility above a floor and
       prices that actually move. A pegged currency (AUD and NZD in the early
       1970s) has near-zero volatility and would otherwise receive enormous
       leverage right before a devaluation.
    2. Risk budget: each asset class gets an equal share of risk, split equally
       across its tradable assets; within the budget the position is inversely
       proportional to the asset's ex-ante volatility.
    3. The whole book is scaled to the target volatility using the ex-ante
       covariance matrix, then per-asset and gross leverage limits are applied.
    """
    assets = signal.index
    tradable = (
        signal.notna()
        & asset_vol.notna()
        & (asset_vol >= cfg.min_asset_vol)
        & (history_days >= cfg.min_history_days)
        & (stale_share.fillna(1.0) <= cfg.max_stale_share)
    )
    w = pd.Series(0.0, index=assets)
    if not tradable.any():
        return w

    sig = signal.where(tradable, 0.0)
    if cfg.long_only:
        sig = tradable.astype(float)

    classes = pd.Series({a: asset_class[a] for a in assets})
    n_in_class = tradable.groupby(classes).sum()
    budget = pd.Series(
        {a: cfg.class_risk_budget[classes[a]] / n_in_class[classes[a]] if tradable[a] else 0.0
         for a in assets}
    )
    raw = sig * budget / asset_vol.where(tradable, np.inf)
    raw = raw.fillna(0.0)

    port_var = float(raw.to_numpy() @ cov @ raw.to_numpy())
    if port_var <= 0:
        return w
    w = raw * (cfg.target_vol / np.sqrt(port_var))

    w = w.clip(-cfg.max_abs_weight, cfg.max_abs_weight)
    gross = w.abs().sum()
    if gross > cfg.max_gross_leverage:
        w *= cfg.max_gross_leverage / gross
    return w


def cost_rates(columns, cost_bps: Mapping[str, float], asset_class: Mapping[str, str] = ASSET_CLASS
               ) -> Dict[str, float]:
    """One-way cost per unit of notional traded, as a decimal."""
    return {c: cost_bps[asset_class[c]] / 1e4 for c in columns}
