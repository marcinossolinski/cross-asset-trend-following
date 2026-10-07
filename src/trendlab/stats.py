"""Statistical inference for backtests: is the Sharpe ratio real?

References
----------
* Newey & West (1987) - heteroskedasticity and autocorrelation consistent errors.
* Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier" - PSR.
* Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio" - DSR.
* Politis & Romano (1994) - stationary bootstrap.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def newey_west_tstat(monthly: pd.Series, lags: int = 12) -> tuple[float, float]:
    """t-statistic and p-value of the mean monthly return with HAC standard errors."""
    y = monthly.dropna().to_numpy()
    res = sm.OLS(y, np.ones_like(y)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def probabilistic_sharpe_ratio(monthly: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark), adjusting for sample length, skew and fat tails.

    Both Sharpe ratios are per period (monthly), not annualised.
    """
    r = monthly.dropna()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = st.skew(r)
    g4 = st.kurtosis(r, fisher=False)          # raw kurtosis (normal = 3)
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2)
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """Expected maximum of N per-period Sharpe ratios under the null of zero skill."""
    srs = np.asarray(trial_sharpes, dtype=float)
    n = len(srs)
    sd = srs.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * st.norm.ppf(1 - 1 / n)
                       + EULER_GAMMA * st.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe_ratio(monthly: pd.Series, trial_monthly_sharpes: Sequence[float]) -> float:
    """PSR against the Sharpe you would expect from the best of N unskilled trials."""
    return probabilistic_sharpe_ratio(monthly, expected_max_sharpe(trial_monthly_sharpes))


def stationary_bootstrap_sharpe(monthly: pd.Series, n_boot: int = 5000, mean_block: int = 6,
                                seed: int = 7) -> np.ndarray:
    """Bootstrap distribution of the annualised Sharpe ratio preserving autocorrelation."""
    r = monthly.dropna().to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        new_block = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
        x = r[idx]
        out[b] = x.mean() / x.std(ddof=1) * np.sqrt(12)
    return out


def alpha_beta(strategy_monthly: pd.Series, factor_monthly: pd.Series, lags: int = 12) -> dict:
    """Regress strategy excess returns on a factor; HAC t-statistics."""
    df = pd.concat([strategy_monthly, factor_monthly], axis=1, join="inner").dropna()
    y, x = df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])
    res = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_ann": float(res.params.iloc[0] * 12),
        "alpha_t": float(res.tvalues.iloc[0]),
        "beta": float(res.params.iloc[1]),
        "beta_t": float(res.tvalues.iloc[1]),
        "r2": float(res.rsquared),
    }
