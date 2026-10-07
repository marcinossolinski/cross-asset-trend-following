"""Static figures for the research note (matplotlib, one consistent style)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical palette (fixed order) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.titlepad": 24, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "ytick.left": False, "font.size": 10.5, "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
    })


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def equity_curves(series: Dict[str, pd.Series], path: Path, title: str, subtitle: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = [BLUE, AQUA, ORANGE]
    for (name, daily), color in zip(series.items(), colors):
        wealth = (1 + daily.fillna(0)).cumprod()
        ax.plot(wealth.index, wealth.values, color=color, label=name,
                lw=2.2 if color == BLUE else 1.5, zorder=3 if color == BLUE else 2)
        ax.annotate(f"{name}  {wealth.iloc[-1]:.0f}x", (wealth.index[-1], wealth.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_title(title)
    _subtitle(ax, subtitle)
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    _save(fig, path)


def drawdowns(series: Dict[str, pd.Series], path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for (name, daily), color in zip(series.items(), [BLUE, ORANGE]):
        w = (1 + daily.fillna(0)).cumprod()
        dd = w / w.cummax() - 1
        ax.plot(dd.index, dd.values, color=color, lw=1.3, label=f"{name} (max {dd.min():.0%})",
                zorder=3 if color == BLUE else 2)
        if color == BLUE:
            ax.fill_between(dd.index, dd.values, 0, color=BLUE, alpha=0.12, lw=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_title("Drawdowns")
    _subtitle(ax, "Peak-to-trough loss of each strategy, daily")
    ax.legend(loc="lower left")
    ax.margins(x=0.01)
    _save(fig, path)


def crisis_bars(ep: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    labels = [f"{pd.Timestamp(p).year}" if pd.Timestamp(p).year == pd.Timestamp(t).year
              else f"{pd.Timestamp(p).year}–{pd.Timestamp(t):%y}" for p, t in zip(ep.peak, ep.trough)]
    x = np.arange(len(ep))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, ep.us_equities, w, color=ORANGE, label="US equities (total return)")
    b2 = ax.bar(x + w / 2 + 0.01, ep.trend_strategy, w, color=BLUE, label="Trend strategy")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:+.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.margins(y=0.12)
    ax.set_title("Crisis behaviour: every US equity drawdown deeper than 15%")
    _subtitle(ax, "Return over each peak-to-trough window, detected mechanically from the data")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def smile_plot(df: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df.equity, df.strategy, s=22, color=BLUE, alpha=0.55, edgecolor=SURFACE, lw=0.6)
    c = np.polyfit(df.equity, df.strategy, 2)
    xs = np.linspace(df.equity.min(), df.equity.max(), 200)
    ax.plot(xs, np.polyval(c, xs), color=INK, lw=1.6)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axvline(0, color=INK2, lw=0.7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("US equity excess return, quarter")
    ax.set_ylabel("Trend strategy return, quarter")
    ax.set_title("The trend-following smile")
    _subtitle(ax, f"Quarterly returns with quadratic fit (curvature {c[0]:.2f})")
    _save(fig, path)


def heatmap(tab: pd.DataFrame, path: Path, baseline_sharpe: float) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    v = tab.to_numpy(dtype=float)
    im = ax.imshow(v, cmap=SEQ_BLUE, aspect="auto", vmin=0, vmax=max(1.0, np.nanmax(v)))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v[i, j] > 0.5 else INK)
    ax.set_xticks(range(tab.shape[1]), [f"{c} days" for c in tab.columns])
    ax.set_yticks(range(tab.shape[0]), [f"{r} month" + ("s" if r > 1 else "") for r in tab.index])
    ax.set_xlabel("Volatility estimation window (EWMA centre of mass)")
    ax.set_ylabel("Momentum look-back")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Sharpe ratio", color=INK2)
    ax.set_title("Parameter robustness")
    _subtitle(ax, f"Sharpe of single-look-back variants (baseline blend: {baseline_sharpe:.2f})")
    _save(fig, path)


def rolling_sharpe(monthly: pd.Series, path: Path, window: int = 60) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    rs = monthly.rolling(window).mean() / monthly.rolling(window).std() * np.sqrt(12)
    ax.plot(rs.index, rs.values, color=BLUE, lw=1.8)
    ax.fill_between(rs.index, rs.values, 0, where=rs.values < 0, color="#e34948", alpha=0.15, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(monthly.mean() / monthly.std() * np.sqrt(12), color=INK2, lw=1, ls="--")
    ax.set_title("Rolling 5-year Sharpe ratio")
    _subtitle(ax, "Dashed line: full-sample Sharpe. Trend following has long flat stretches.")
    ax.margins(x=0.01)
    _save(fig, path)


def class_contributions(contrib: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ends = {}
    for col, color in zip(["Equity", "FX", "Commodity"], [BLUE, ORANGE, AQUA]):
        if col in contrib:
            cum = contrib[col].fillna(0).cumsum()
            ax.plot(cum.index, cum.values, color=color, label=col)
            ends[col] = (cum.index[-1], cum.iloc[-1])
    # direct labels, pushed apart so they never collide
    order = sorted(ends, key=lambda k: ends[k][1])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed = []
    for col in order:
        y = ends[col][1]
        if placed and y - placed[-1] < 0.06 * span:
            y = placed[-1] + 0.06 * span
        placed.append(y)
        ax.annotate(col, (ends[col][0], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_title("Where the returns come from")
    _subtitle(ax, "Cumulative P&L contribution by asset class (sum of daily returns, before costs)")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    _save(fig, path)


def averaging_bias_plot(df: pd.DataFrame, path: Path) -> None:
    _style()
    piv = df.pivot(index="asset", columns="price_series", values="trend_1m_sharpe")
    piv = piv.sort_values("month_end")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(piv))
    h = 0.38
    ax.barh(y + h / 2 + 0.01, piv["monthly_average"], h, color=ORANGE, label="Monthly-average prices (leaky)")
    ax.barh(y - h / 2 - 0.01, piv["month_end"], h, color=BLUE, label="Month-end prices (point-in-time)")
    ax.set_yticks(y, piv.index)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Sharpe ratio of a 1-month trend signal")
    ax.set_title("Why monthly-average data was excluded")
    _subtitle(ax, "Same assets, same rule: averaging prices manufactures momentum that was never tradable")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
