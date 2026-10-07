# Data sources

| File / series | Source | Licence | Notes |
|---|---|---|---|
| US equity total return, risk-free return | [SteelCerberus/us-market-data](https://github.com/SteelCerberus/us-market-data) | MIT | S&P 500 with dividends reinvested; T-bill based risk-free index |
| FX rates (AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK) | Federal Reserve H.10, via [datasets/exchange-rates](https://github.com/datasets/exchange-rates) | Public domain (PDDL) | Quoted as foreign currency per USD; converted to the return of holding the foreign currency |
| WTI and Brent spot prices | U.S. EIA, via [datasets/oil-prices](https://github.com/datasets/oil-prices) | Public domain (PDDL) | Daily spot; the negative WTI print of 2020-04-20 is treated as missing |

`processed/daily_excess_returns.csv` is the exact panel used for every result in this repository
(January 1971 – November 2025). Rebuild it with `python scripts/build_dataset.py --force`; newer
downloads will extend the sample, so results may differ slightly from the published ones.

Raw downloads are cached in `data/raw/`, which is not committed.
