"""Ex-ante risk estimates computed only from data available at each date."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 256  # the union FX/equity/commodity calendar has ~256 business days a year


def ewma_vol(returns: pd.DataFrame, com_days: int) -> pd.DataFrame:
    """Annualised EWMA volatility (zero-mean), as in Moskowitz, Ooi & Pedersen (2012).

    The value on date t uses returns up to and including t.
    """
    var = (returns ** 2).ewm(com=com_days, min_periods=com_days, ignore_na=True).mean()
    return np.sqrt(var * TRADING_DAYS)


def ewma_cov_at(returns: pd.DataFrame, dates: pd.DatetimeIndex, halflife_days: int) -> dict:
    """Annualised EWMA covariance matrices evaluated at ``dates``.

    Recursion S_t = lam * S_{t-1} + (1 - lam) * r_t r_t', with missing returns
    treated as zero (the asset is not live yet). Returns {date: ndarray}.
    """
    lam = 0.5 ** (1.0 / halflife_days)
    x = returns.fillna(0.0).to_numpy()
    wanted = set(dates)
    n = x.shape[1]
    s = np.zeros((n, n))
    weight = 0.0
    out = {}
    for t, (d, r) in enumerate(zip(returns.index, x)):
        s = lam * s + (1.0 - lam) * np.outer(r, r)
        weight = lam * weight + (1.0 - lam)
        if d in wanted:
            out[d] = s / weight * TRADING_DAYS       # bias-corrected for the finite history
    return out
