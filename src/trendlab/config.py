"""Universe definition, data sources and strategy parameters.

Everything that a researcher might want to change lives here, so that the
research script never contains hidden magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# Data sources (public, point-in-time daily closes)
# --------------------------------------------------------------------------- #
RAW_URLS: Dict[str, str] = {
    # US equity total return index + risk-free return index (daily, MIT licence)
    "us_market": "https://raw.githubusercontent.com/SteelCerberus/us-market-data/main/data/us_market_data.csv",
    # Federal Reserve H.10 daily FX rates, quoted as foreign currency per USD
    "fx": "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv",
    # EIA daily spot prices
    "wti": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv",
    "brent": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv",
}

# Federal Reserve H.10 country names -> asset code (long = long foreign currency vs USD)
FX_COUNTRIES: Dict[str, str] = {
    "Australia": "AUD",
    "Canada": "CAD",
    "Switzerland": "CHF",
    "Euro": "EUR",
    "United Kingdom": "GBP",
    "Japan": "JPY",
    "Norway": "NOK",
    "New Zealand": "NZD",
    "Sweden": "SEK",
}

ASSET_CLASS: Dict[str, str] = {
    "US_EQ": "Equity",
    **{code: "FX" for code in FX_COUNTRIES.values()},
    "WTI": "Commodity",
    "BRENT": "Commodity",
}

# One-way transaction cost per unit of notional traded, in basis points.
# Deliberately conservative versus liquid futures / FX forwards.
COST_BPS: Dict[str, float] = {
    "Equity": 3.0,
    "FX": 3.0,
    "Commodity": 10.0,
}

SAMPLE_START = "1971-01-01"  # data start (used to warm up signals and risk estimates)
SAMPLE_END = "2025-11-30"    # last full month covered by every core source


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the trend-following strategy.

    The baseline follows Hurst, Ooi & Pedersen (2017): an equal-weighted blend
    of 1-, 3- and 12-month time-series momentum, with positions sized by
    ex-ante volatility. It is fixed *a priori* from the literature and is not
    the result of a parameter search on this data set.
    """

    lookbacks: Tuple[int, ...] = (1, 3, 12)      # months
    signal: str = "sign"                          # "sign" or "scaled"
    vol_com_days: int = 60                        # EWMA centre of mass for asset vol
    cov_halflife_days: int = 126                  # EWMA half-life for the covariance matrix
    min_history_days: int = 260                   # days of history before an asset can trade
    min_asset_vol: float = 0.01                   # tradability: skip pegged markets (near-zero vol)
    max_stale_share: float = 0.5                  # tradability: skip if >50% of last 63 days had no price change
    target_vol: float = 0.10                      # annualised portfolio volatility target
    max_gross_leverage: float = 5.0
    max_abs_weight: float = 2.0
    class_risk_budget: Dict[str, float] = field(
        default_factory=lambda: {"Equity": 1 / 3, "FX": 1 / 3, "Commodity": 1 / 3}
    )
    cost_multiplier: float = 1.0
    # First allowed rebalance. Fixed exchange rates (Bretton Woods / Smithsonian)
    # ended in March 1973; before that, currencies were administered rather than
    # freely traded, so positions only start once the floating regime is in place.
    trading_start: str = "1973-12-31"
    long_only: bool = False                       # True -> risk-balanced buy-and-hold benchmark

    def with_(self, **kwargs) -> "StrategyConfig":
        return replace(self, **kwargs)

    @property
    def label(self) -> str:
        lb = "+".join(str(x) for x in self.lookbacks)
        return f"TSMOM[{lb}m|{self.signal}|com{self.vol_com_days}]"


BASELINE = StrategyConfig()
