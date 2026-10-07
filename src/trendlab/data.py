"""Download raw public data and build a point-in-time panel of daily excess returns.

Design rules
------------
* Only series observed as **daily closes** are used. Monthly *averages*
  (common for gold and Treasury yields in free data sets) are excluded on
  purpose: returns computed from averaged prices overlap with information that
  is already known at the rebalance date, which creates spurious momentum
  (see ``analysis.averaging_bias``).
* Each series is converted to returns on its **own** trading calendar and only
  then aligned to the union calendar. Days on which a market is closed get a
  zero return, so multi-day moves are booked on the next trading day and never
  earlier.
* Returns are excess returns, i.e. what a futures / forward position would
  earn before carry: equity total return minus the risk-free return, FX and
  oil spot returns (carry and roll yield are not available point-in-time in
  free data and are therefore ignored; see README "Limitations").
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import ASSET_CLASS, FX_COUNTRIES, RAW_URLS, SAMPLE_END, SAMPLE_START


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_raw(raw_dir: Path, force: bool = False) -> Dict[str, Path]:
    """Download every raw file into ``raw_dir`` (cached unless ``force``)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, url in RAW_URLS.items():
        path = raw_dir / f"{name}.csv"
        if force or not path.exists():
            print(f"downloading {name:10s} <- {url}")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


# --------------------------------------------------------------------------- #
# Series -> returns on their own calendar
# --------------------------------------------------------------------------- #
def _clean_prices(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Non-positive prices (e.g. WTI spot on 2020-04-20) cannot produce a
    # meaningful percentage return; treat them as a missing observation.
    s = s.where(s > 0)
    return s.dropna()


def us_equity_excess(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    tr = _clean_prices(df["Adjusted Close"])          # total return index (dividends reinvested)
    rf = _clean_prices(df["Risk Free Return"])        # cumulative risk-free index
    idx = tr.index.intersection(rf.index)
    ex = tr.loc[idx].pct_change() - rf.loc[idx].pct_change()
    return ex.rename("US_EQ")


def us_equity_total_return(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return _clean_prices(df["Adjusted Close"]).pct_change().rename("US_EQ_TR")


def fx_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    out = {}
    for country, code in FX_COUNTRIES.items():
        s = df.loc[df["Country"] == country].set_index("Date")["Exchange rate"]
        rate = _clean_prices(s)                       # foreign currency per USD
        out[code] = rate.shift(1) / rate - 1.0        # return of holding 1 unit of foreign currency
    return pd.DataFrame(out)


def spot_returns(path: Path, name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return _clean_prices(df["Price"]).pct_change().rename(name)


def spot_prices(path: Path, name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return _clean_prices(df["Price"]).rename(name)


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def _align(native: Dict[str, pd.Series], start: str, end: str) -> pd.DataFrame:
    """Align returns computed on native calendars to the union business-day calendar."""
    cal = pd.DatetimeIndex(sorted(set().union(*[s.dropna().index for s in native.values()])))
    cal = cal[(cal.dayofweek < 5) & (cal >= start) & (cal <= end)]
    cols = {}
    for name, s in native.items():
        s = s.dropna()
        s = s[(s.index >= start) & (s.index <= end)]
        aligned = s.reindex(cal)
        live = cal >= s.index.min()
        aligned[live & aligned.isna()] = 0.0         # market closed -> no move booked today
        cols[name] = aligned
    return pd.DataFrame(cols, index=cal)


def build_daily_excess_returns(raw: Dict[str, Path], start: str = SAMPLE_START,
                               end: str = SAMPLE_END) -> pd.DataFrame:
    native: Dict[str, pd.Series] = {"US_EQ": us_equity_excess(raw["us_market"])}
    for code, s in fx_returns(raw["fx"]).items():
        native[code] = s
    native["WTI"] = spot_returns(raw["wti"], "WTI")
    native["BRENT"] = spot_returns(raw["brent"], "BRENT")
    panel = _align(native, start, end)
    panel = panel[list(ASSET_CLASS)]                  # fixed column order
    panel.index.name = "date"
    return panel


def build_benchmark_total_return(raw: Dict[str, Path], index: pd.DatetimeIndex) -> pd.Series:
    """US equity total return (not excess) on the panel calendar, for drawdown episodes."""
    s = us_equity_total_return(raw["us_market"])
    out = _align({"US_EQ_TR": s}, str(index.min().date()), str(index.max().date()))["US_EQ_TR"]
    return out.reindex(index).fillna(0.0)


def quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple data-integrity checks that a reviewer would ask for."""
    rows = []
    for c in panel.columns:
        s = panel[c].dropna()
        zero = (s == 0).astype(int)
        longest_flat = int(zero.groupby((zero != zero.shift()).cumsum()).sum().max())
        rows.append({
            "asset": c,
            "asset_class": ASSET_CLASS[c],
            "first_date": s.index.min().date(),
            "last_date": s.index.max().date(),
            "observations": len(s),
            "ann_vol": s.std() * np.sqrt(252),
            "share_zero_days": (s == 0).mean(),
            "longest_flat_run_days": longest_flat,
            "min_daily_return": s.min(),
            "max_daily_return": s.max(),
            "days_abs_ret_gt_15pct": int((s.abs() > 0.15).sum()),
        })
    return pd.DataFrame(rows).set_index("asset")


def load_panel(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)
