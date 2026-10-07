"""Run the full research pipeline and write every table and figure in results/.

Usage:
    python scripts/run_research.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendlab import analysis, plotting  # noqa: E402
from trendlab.backtest import Precomputed, run_backtest  # noqa: E402
from trendlab.config import BASELINE  # noqa: E402
from trendlab.data import load_panel  # noqa: E402
from trendlab.metrics import sharpe, summary, to_monthly  # noqa: E402
from trendlab.stats import (  # noqa: E402
    alpha_beta,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_sharpe,
)

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    R = load_panel(ROOT / "data" / "processed" / "daily_excess_returns.csv")
    eq_tr = load_panel(ROOT / "data" / "processed" / "us_equity_total_return.csv")["US_EQ_TR"]
    pre = Precomputed(R)

    # ------------------------------------------------------------------ core
    trend = run_backtest(R, BASELINE, pre)
    long_only = run_backtest(R, BASELINE.with_(long_only=True), pre)
    start = trend.returns.index.min()
    us_eq = R["US_EQ"].loc[start:]

    perf = pd.DataFrame({
        "Trend (net of costs)": summary(trend.returns, trend.turnover, trend.costs),
        "Trend (before costs)": summary(trend.gross_returns),
        "Long-only, same risk budgets": summary(long_only.returns, long_only.turnover, long_only.costs),
        "US equities (excess return)": summary(us_eq),
    })
    perf.to_csv(TAB / "performance_summary.csv")

    # ------------------------------------------------------------ inference
    m = trend.monthly
    grid_cfgs = analysis.parameter_grid(BASELINE)
    grid = analysis.run_grid(R, grid_cfgs, pre)
    trial_sr_monthly = [r.monthly.mean() / r.monthly.std() for r in grid.values()]
    boot = stationary_bootstrap_sharpe(m)
    nw_t, nw_p = newey_west_tstat(m)
    inference = {
        "n_months": int(m.count()),
        "sharpe": sharpe(m),
        "newey_west_t": nw_t,
        "newey_west_p": nw_p,
        "psr_vs_zero": probabilistic_sharpe_ratio(m, 0.0),
        "n_trials": len(grid_cfgs),
        "expected_max_sharpe_of_trials_ann": expected_max_sharpe(trial_sr_monthly) * np.sqrt(12),
        "deflated_sharpe_ratio": deflated_sharpe_ratio(m, trial_sr_monthly),
        "bootstrap_sharpe_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "bootstrap_p_sharpe_le_0": float((boot <= 0).mean()),
        "vs_us_equities": alpha_beta(m, to_monthly(us_eq)),
        "vs_long_only": alpha_beta(m, long_only.monthly),
    }

    # ------------------------------------------------------------ robustness
    full = run_backtest(R, BASELINE.with_(trading_start="1971-01-01"), pre)
    sample_start = {
        "first_month": f"{full.monthly.index.min():%Y-%m}",
        "sharpe": sharpe(full.monthly),
        "excess_kurtosis": float(full.monthly.kurt()),
        "best_month": f"{full.monthly.idxmax():%Y-%m} ({full.monthly.max():+.1%})",
    }
    lbs, coms = (1, 3, 6, 9, 12, 18, 24), (20, 60, 120)
    sens = analysis.sensitivity_table(grid, lbs, coms)
    sens.to_csv(TAB / "sensitivity_sharpe.csv")
    grid_table = pd.DataFrame({k: summary(v.returns, v.turnover, v.costs) for k, v in grid.items()}).T
    grid_table.to_csv(TAB / "all_trials.csv")

    oos, wf_log = analysis.walk_forward(grid)
    wf_log.to_csv(TAB / "walk_forward_selection.csv")
    same_period = m.loc[oos.index]
    walk_forward = {
        "period": f"{oos.index.min():%Y-%m} to {oos.index.max():%Y-%m}",
        "oos_sharpe_walk_forward": sharpe(oos),
        "baseline_sharpe_same_period": sharpe(same_period),
        "times_selection_changed": int((wf_log.selected != wf_log.selected.shift()).sum() - 1),
        "most_selected": wf_log.selected.value_counts().head(3).to_dict(),
    }

    costs = analysis.cost_sensitivity(R, BASELINE, pre)
    costs.to_csv(TAB / "cost_sensitivity.csv")

    decades = pd.DataFrame({
        "trend": analysis.subperiod_sharpe(m),
        "long_only": analysis.subperiod_sharpe(long_only.monthly),
        "us_equities": analysis.subperiod_sharpe(to_monthly(us_eq)),
    })
    decades.to_csv(TAB / "sharpe_by_decade.csv")

    # ------------------------------------------------------------ crisis
    episodes = analysis.equity_drawdown_episodes(eq_tr, trend.returns)
    episodes.to_csv(TAB / "equity_drawdown_episodes.csv", index=False)
    sm = analysis.smile(trend.returns, us_eq)
    curvature = float(np.polyfit(sm.equity, sm.strategy, 2)[0])

    # ------------------------------------------------------------ attribution
    contrib = trend.class_contributions()
    class_pnl = (contrib.sum() / (len(m) / 12)).rename("avg_annual_contribution")
    class_pnl.to_csv(TAB / "asset_class_contribution.csv")
    corr = pd.concat({"trend": m, "us_equities": to_monthly(us_eq),
                      "long_only": long_only.monthly}, axis=1).corr()
    corr.to_csv(TAB / "correlations.csv")

    # ------------------------------------------------------------ data integrity
    bias = analysis.averaging_bias(R)
    bias.to_csv(TAB / "averaging_bias.csv", index=False)
    bias_avg = bias.groupby("price_series")[["lag1_autocorr", "trend_1m_sharpe"]].mean()

    # ------------------------------------------------------------ figures
    plotting.equity_curves(
        {"Trend": trend.returns, "Long-only": long_only.returns, "US equities": us_eq},
        FIG / "equity_curve.png",
        "Growth of $1, excess of cash",
        f"{start:%b %Y} - {m.index.max():%b %Y}, log scale. Trend and long-only target 10% volatility; "
        "equities are unlevered.")
    plotting.drawdowns({"Trend": trend.returns, "US equities": us_eq}, FIG / "drawdowns.png")
    plotting.crisis_bars(episodes, FIG / "crisis_alpha.png")
    plotting.smile_plot(sm, FIG / "smile.png")
    plotting.heatmap(sens, FIG / "sensitivity_heatmap.png", sharpe(m))
    plotting.rolling_sharpe(m, FIG / "rolling_sharpe.png")
    plotting.class_contributions(contrib, FIG / "asset_class_pnl.png")
    plotting.averaging_bias_plot(bias, FIG / "averaging_bias.png")

    # ------------------------------------------------------------ save + print
    out = {
        "baseline": BASELINE.label,
        "sample": f"{m.index.min():%Y-%m} to {m.index.max():%Y-%m}",
        "performance": perf.to_dict(),
        "inference": inference,
        "walk_forward": walk_forward,
        "including_bretton_woods_exit": sample_start,
        "smile_curvature": curvature,
        "averaging_bias": bias_avg.to_dict(),
        "asset_class_contribution": class_pnl.to_dict(),
        "correlations": corr.to_dict(),
    }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2, default=str))

    pd.set_option("display.width", 160)
    print(perf.to_string())
    print(json.dumps(inference, indent=2))
    print(json.dumps(walk_forward, indent=2))
    print(json.dumps(sample_start, indent=2))
    print(costs.round(3).to_string())
    print(decades.round(2).to_string())
    print(episodes.to_string())
    print("smile curvature", round(curvature, 3))
    print(bias_avg.round(3).to_string())
    print(class_pnl.round(4).to_string())
    print(corr.round(2).to_string())
    print(sens.round(2).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
