"""Time-series momentum signals, observed at month-end rebalance dates."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of every calendar month in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def wealth_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative growth of 1 unit, NaN before an asset's first observation."""
    return (1.0 + returns).cumprod()


def trailing_returns(returns: pd.DataFrame, dates: pd.DatetimeIndex, months: int) -> pd.DataFrame:
    """Excess return over the past ``months`` months, known at the close of each date.

    Only information up to and including the rebalance date is used.
    """
    w = wealth_index(returns).loc[dates]
    return w / w.shift(months) - 1.0


def tsmom_signal(returns: pd.DataFrame, dates: pd.DatetimeIndex, lookbacks: Sequence[int],
                 kind: str = "sign", ann_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Equal-weighted blend of trend signals over several look-back windows.

    kind="sign"   : Moskowitz, Ooi & Pedersen (2012) / Hurst, Ooi & Pedersen (2017),
                    the sign of the past excess return, in {-1, 0, +1}.
    kind="scaled" : risk-adjusted strength, past return divided by its expected
                    standard deviation and clipped to [-1, 1]. Needs ``ann_vol``.
    An asset gets NaN (not traded) unless every look-back is available.
    """
    parts = []
    for m in lookbacks:
        r = trailing_returns(returns, dates, m)
        if kind == "sign":
            parts.append(np.sign(r))
        elif kind == "scaled":
            if ann_vol is None:
                raise ValueError("kind='scaled' requires ann_vol")
            z = r / (ann_vol.loc[dates] * np.sqrt(m / 12.0))
            parts.append((z / 2.0).clip(-1.0, 1.0))
        else:
            raise ValueError(f"unknown signal kind: {kind}")
    stacked = np.stack([p.to_numpy() for p in parts])
    blended = np.mean(stacked, axis=0)                 # NaN if any look-back is missing
    return pd.DataFrame(blended, index=dates, columns=returns.columns)
