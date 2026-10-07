import numpy as np
import pandas as pd
import pytest

from trendlab.backtest import Precomputed, run_backtest
from trendlab.config import COST_BPS, StrategyConfig

CFG = StrategyConfig(trading_start="2000-01-01")


def test_no_lookahead(synthetic_returns):
    """Changing the future must not change any past decision or past return."""
    base = run_backtest(synthetic_returns, CFG)

    cutoff = pd.Timestamp("2004-06-30")
    shocked = synthetic_returns.copy()
    future = shocked.index > cutoff
    rng = np.random.default_rng(0)
    shocked.loc[future] = rng.normal(0, 0.05, size=shocked.loc[future].shape)  # wild, different future
    alt = run_backtest(shocked, CFG)

    past_rebal = base.target_weights.index[base.target_weights.index <= cutoff]
    pd.testing.assert_frame_equal(base.target_weights.loc[past_rebal], alt.target_weights.loc[past_rebal])
    pd.testing.assert_series_equal(base.returns.loc[:cutoff], alt.returns.loc[:cutoff])
    # and the future really was different, otherwise the test proves nothing
    assert not np.allclose(base.returns.loc[cutoff:].iloc[1:], alt.returns.loc[cutoff:].iloc[1:])


def test_trade_earns_from_next_day(synthetic_returns):
    """A jump on the rebalance day itself belongs to the old position, not the new one."""
    r = synthetic_returns.copy()
    res = run_backtest(r, CFG)
    d = res.target_weights.index[30]
    old_position = res.contributions.loc[d, "US_EQ"] / r.loc[d, "US_EQ"]   # drifted weight held into d
    assert abs(old_position) > 0

    r.loc[d, "US_EQ"] += 0.20                    # huge move in US_EQ on the rebalance day
    res2 = run_backtest(r, CFG)
    jump_pnl = res2.contributions.loc[d, "US_EQ"] - res.contributions.loc[d, "US_EQ"]
    # the jump is earned by the position held *before* the rebalance ...
    assert jump_pnl == pytest.approx(0.20 * old_position)
    # ... while the position decided at d's close only starts earning on the next day
    nxt = r.index[r.index.get_loc(d) + 1]
    new_w = res2.target_weights.loc[d, "US_EQ"]
    assert res2.contributions.loc[nxt, "US_EQ"] == pytest.approx(new_w * r.loc[nxt, "US_EQ"])


def test_portfolio_hits_volatility_target(synthetic_returns):
    res = run_backtest(synthetic_returns, CFG.with_(long_only=True))
    realised = res.returns.std() * np.sqrt(256)
    assert realised == pytest.approx(CFG.target_vol, rel=0.20)


def test_costs_are_charged_on_traded_notional(synthetic_returns):
    free = run_backtest(synthetic_returns, CFG.with_(cost_multiplier=0.0))
    paid = run_backtest(synthetic_returns, CFG)
    pd.testing.assert_series_equal(free.gross_returns, paid.gross_returns)
    assert (paid.returns <= free.returns + 1e-15).all()
    total_cost = (free.returns - paid.returns).sum()
    assert total_cost == pytest.approx(paid.costs.sum())
    max_rate = max(COST_BPS.values()) / 1e4
    assert paid.costs.sum() <= paid.turnover.sum() * max_rate + 1e-12


def test_pegged_market_is_not_traded(synthetic_returns):
    r = synthetic_returns.copy()
    r["CAD"] = 0.0                               # a frozen / pegged market
    res = run_backtest(r, CFG)
    assert (res.target_weights["CAD"] == 0).all()


def test_leverage_limits(synthetic_returns):
    cfg = CFG.with_(target_vol=0.60, max_gross_leverage=1.5, max_abs_weight=0.8)
    w = run_backtest(synthetic_returns, cfg).target_weights
    assert (w.abs().sum(axis=1) <= 1.5 + 1e-9).all()
    assert (w.abs() <= 0.8 + 1e-9).all().all()


def test_precomputed_cache_gives_identical_results(synthetic_returns):
    pre = Precomputed(synthetic_returns)
    a = run_backtest(synthetic_returns, CFG, pre)
    b = run_backtest(synthetic_returns, CFG)
    pd.testing.assert_series_equal(a.returns, b.returns)
