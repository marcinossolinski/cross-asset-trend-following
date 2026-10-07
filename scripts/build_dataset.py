"""Download public data and build the processed daily excess-return panel.

Usage:
    python scripts/build_dataset.py            # uses cached raw files if present
    python scripts/build_dataset.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trendlab.data import (  # noqa: E402
    build_benchmark_total_return,
    build_daily_excess_returns,
    download_raw,
    quality_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download raw files")
    args = parser.parse_args()

    raw = download_raw(ROOT / "data" / "raw", force=args.force)
    panel = build_daily_excess_returns(raw)
    bench = build_benchmark_total_return(raw, panel.index)

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    panel.round(8).to_csv(out / "daily_excess_returns.csv")
    bench.round(8).to_frame().to_csv(out / "us_equity_total_return.csv")

    report = quality_report(panel)
    report.to_csv(out / "data_quality_report.csv")
    print(report.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\npanel: {panel.shape[0]:,} days x {panel.shape[1]} assets, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")


if __name__ == "__main__":
    main()
