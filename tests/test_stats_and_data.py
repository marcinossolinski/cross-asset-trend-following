import numpy as np
import pandas as pd
import pytest

from trendlab.analysis import averaging_bias, equity_drawdown_episodes
from trendlab.data import fx_returns
from trendlab.metrics import drawdown, sharpe
from trendlab.signals import rebalance_dates, tsmom_signal
from trendlab.stats import expected_max_sharpe, probabilistic_sharpe_ratio


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
    assert drawdown(r).min() == pytest.approx(-0.50)


def test_sharpe_formula():
    m = pd.Series([0.01, 0.03, -0.01, 0.02], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    assert sharpe(m) == pytest.approx(m.mean() / m.std() * np.sqrt(12))


def test_psr_is_one_half_at_its_own_sharpe():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0.01, 0.04, 240))
    sr = m.mean() / m.std()
    assert probabilistic_sharpe_ratio(m, sr) == pytest.approx(0.5, abs=1e-9)
    assert probabilistic_sharpe_ratio(m, 0.0) > 0.5


def test_more_trials_raise_the_bar():
    rng = np.random.default_rng(2)
    few = expected_max_sharpe(rng.normal(0, 0.05, 5))
    many = expected_max_sharpe(rng.normal(0, 0.05, 500))
    assert many > few > 0


def test_fx_returns_long_foreign_currency(tmp_path):
    # foreign currency per USD halves -> foreign currency doubled in USD terms
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Country": ["Japan", "Japan"],
                       "Exchange rate": [2.0, 1.0]})
    f = tmp_path / "fx.csv"
    df.to_csv(f, index=False)
    r = fx_returns(f)["JPY"].dropna()
    assert r.iloc[0] == pytest.approx(1.0)


def test_signal_uses_only_past_prices():
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    r = pd.DataFrame({"US_EQ": 0.001}, index=idx)
    dates = rebalance_dates(idx)
    s1 = tsmom_signal(r, dates, (12,))
    r2 = r.copy()
    r2.loc["2020-07-01":] = -0.01
    s2 = tsmom_signal(r2, dates, (12,))
    pd.testing.assert_frame_equal(s1.loc[:"2020-06-30"], s2.loc[:"2020-06-30"])


def test_drawdown_episode_detection():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(0.0, index=idx)
    r.iloc[50:70] = -0.012          # ~21% fall
    r.iloc[70:200] = 0.003          # recovery to a new high
    episodes = equity_drawdown_episodes(r, pd.Series(0.0, index=idx), threshold=0.15)
    assert len(episodes) == 1
    assert episodes.loc[0, "us_equities"] < -0.15


def test_averaging_creates_spurious_autocorrelation():
    """Working (1960): averages of a random walk have ~0.25 first-order autocorrelation."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("1960-01-01", "2019-12-31")
    r = pd.DataFrame({"X": rng.normal(0, 0.01, len(idx))}, index=idx)
    res = averaging_bias(r).set_index("price_series")
    assert abs(res.loc["month_end", "lag1_autocorr"]) < 0.10
    assert res.loc["monthly_average", "lag1_autocorr"] > 0.15
