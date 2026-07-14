---
name: macro-shock-risk-parity
description: Backtest and run a multi-asset risk-parity allocation model (Asset RP / Macro RP / Joint RP) across CSI 300, CSI 500, HSI, S&P 500, Chinese government/credit bonds, convertibles, gold, and commodities, with a six-dimension macro shock overlay (growth/inflation/liquidity/rates/credit/external) extracted via Kalman filtering. Use when the user wants to run, extend, or debug this risk-parity backtest, add/remove assets or macro factors, tune cap/shrinkage/smoothing parameters, or understand why Joint RP over/underperforms in a given period. Chinese keywords: 风险平价、宏观冲击、资产配置回测、Macro RP、Joint RP
---

# Macro Shock Risk Parity

A multi-asset risk-parity backtesting model with a macroeconomic shock overlay. Three strategies share the same asset universe and rebalancing logic, differing only in the covariance matrix used to solve for risk-parity weights:

- **Asset RP** — sample covariance of daily asset returns.
- **Macro RP** — covariance implied by each asset's rolling regression exposure to six macro shock factors.
- **Joint RP** — trace-normalized blend of the two covariances above, with additional weight smoothing.

## When to use this skill

Invoke it when the user wants to:
- Run the full backtest (`scripts/main.py`) and produce a performance table + charts.
- Add/remove assets in the universe, or add/remove macro indicators.
- Tune model parameters (rolling window, shrinkage, cap levels, smoothing) in `scripts/config.py`.
- Understand or debug the pipeline: data fetching → macro z-scoring/orthogonalization → Kalman shock extraction → covariance construction → risk-parity solve → low-vol cap → drift/rebalance simulation.
- Investigate why one strategy (esp. Joint RP) over- or under-performs in a given sample period — see `references/joint_rp_diagnosis_notes.md` for a worked example of this kind of diagnosis.

## Pipeline (in execution order)

1. **`data.py`** — fetches asset prices and macro indicators via [akshare](https://github.com/akfamily/akshare) (free, public data), converts non-CNY assets to RMB terms, caches every raw fetch as local CSV so re-runs don't hit the network. All fetch functions are keyed by `source` string in `config.ASSET_UNIVERSE` / macro group specs — adding a new source means adding a `_fetch_*` function and registering it in `_ASSET_FETCHERS` / `_MACRO_FETCHERS`.
2. **`macro_factors.py`** — for each of the six macro groups, pulls each raw indicator, applies its publication lag, computes a rolling (time-window, not row-count) z-score, then Gram-Schmidt-orthogonalizes indicators within a group so a composite (equal-weight mean) doesn't double-count correlated indicators. Everything here respects "point-in-time" availability — no look-ahead.
3. **`shocks.py`** — runs a local-linear-trend Kalman filter over each of the six composite macro signals, but only recomputes on "event days" (days the signal actually changes) rather than every calendar day, to avoid Kalman variance collapsing on the flat forward-filled stretches between releases. Output is a standardized innovation ("shock") per factor per day.
4. **`risk_parity.py`** — core solver: risk-contribution-based iterative weighting (with a ratio clip to prevent oscillation under negatively correlated assets), plus the three covariance constructions (sample / macro-implied / trace-blended joint) and the low-volatility-group cap enforcement.
5. **`backtest.py`** — month-end decision, next-trading-day execution, buy-and-hold drift between rebalances, transaction costs on turnover, and performance/plotting utilities. Strictly no look-ahead: weights at month-end *t* only use data through *t*.
6. **`main.py`** — orchestrates the full run across all (strategy × cap level) combinations and writes `output/performance_table.csv` + PNG charts.

## Running it

```bash
pip install -r requirements.txt
cd scripts
python validate.py   # run after any code change, before trusting backtest output
python main.py
```

First run will fetch and cache data under `scripts/cache/` (gitignored) — this can take a few minutes depending on akshare's upstream sources. Subsequent runs reuse the cache; delete individual CSVs under `cache/` to force a refresh for one series.

If this skill is installed somewhere read-only (e.g. a shared skills directory), set `MSRP_CACHE_DIR` / `MSRP_OUTPUT_DIR` to redirect the cache and output to a writable location instead.

## Key parameters (`scripts/config.py`)

| Parameter | Meaning |
|---|---|
| `ROLLING_WINDOW_DAYS` | Lookback for covariance/regression (default 3y) |
| `SHRINKAGE_INTENSITY` | Shrinkage of macro factor covariance toward diagonal |
| `JOINT_MIX_WEIGHT` | Weight on sample vs. macro-implied covariance in Joint RP |
| `SMOOTHING_LAMBDA` | Joint RP weight smoothing speed (lower = slower, less turnover, more lag) |
| `CAP_LEVELS` | Cap on total weight of the low-volatility asset group, swept per backtest |
| `RP_RATIO_CLIP` | Per-iteration weight adjustment clip, guards convergence under negative correlation |

## Known behavior / gotchas

- Joint RP's smoothing (`SMOOTHING_LAMBDA`) helps in trending, cap-constrained regimes and hurts across long samples with multiple trend reversals or when `cap=1.0` removes the diversification constraint entirely — see `references/joint_rp_diagnosis_notes.md` before assuming a Sharpe gap is a data or code bug.
- Assets with shorter history (e.g. gold, commodity index) are dynamically excluded from a given rebalance if they don't yet have a full rolling window — this is intentional, not a bug.
- 南华商品指数 (Nanhua Commodity Index) was delisted from akshare and is proxied by 中证商品期货指数 from 2022-07 onward; check `config.py` comments before "fixing" an apparent discontinuity.
