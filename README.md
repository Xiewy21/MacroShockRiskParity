# Macro Shock Risk Parity

A multi-asset risk-parity backtest with a macroeconomic shock overlay, covering CSI 300, CSI 500, HSI, S&P 500, Chinese government/credit bonds, convertibles, gold, and commodities. Three strategies (Asset RP / Macro RP / Joint RP) share the same rebalancing engine and differ only in which covariance matrix drives the risk-parity solve. See [SKILL.md](SKILL.md) for the full pipeline description, parameters, and known behavior.

## Use as a Claude Code Skill

Clone this repo into your Claude Code skills directory (or a project's `.claude/skills/`) so Claude can pick it up:

```bash
git clone <this-repo-url> ~/.claude/skills/macro-shock-risk-parity
```

Claude will use `SKILL.md` to decide when to invoke it — e.g. when you ask it to run the backtest, tune parameters, add an asset, or explain why one strategy is under/over-performing.

## Run it standalone

```bash
pip install -r requirements.txt
cd scripts
python main.py
```

All market and macro data is fetched on first run via [akshare](https://github.com/akfamily/akshare) (free, public sources) and cached locally under `scripts/cache/` — no data files are bundled in this repo. Outputs (performance table + charts) land in `scripts/output/`. Set `MSRP_CACHE_DIR` / `MSRP_OUTPUT_DIR` to redirect these to a writable location if the repo itself is checked out read-only (e.g. installed as a shared skill).

Last verified with akshare 1.18.64.

## Layout

```
SKILL.md                          # skill definition Claude reads
scripts/                          # the model itself
  config.py                       # asset universe, macro groups, all tunable parameters
  data.py                         # akshare fetch + local CSV cache
  macro_factors.py                # publication-lag-aware z-score + orthogonalization
  shocks.py                       # Kalman-filter shock extraction
  risk_parity.py                  # risk-parity solver + covariance construction
  backtest.py                     # rebalancing simulation + performance/plots
  main.py                         # entry point
references/
  joint_rp_diagnosis_notes.md     # worked example: diagnosing a Sharpe gap across strategies/cap levels
```

## Methodology reference

The six-dimension macro shock overlay and risk-parity blending approach is
based on the following CICC (中金公司) research report:

> 陈宜筠, CFA (SAC S0080524080004 / SFC BTZ190); 周萧潇 (SAC S0080521010006 /
> SFC BRA090); 刘均伟 (SAC S0080520120002 / SFC BQR365). 《量化配置模型（3）：
> 如何应对低利率环境——基于宏观冲击的风险平价模型增强》. CICC Research,
> published 2026-06-23.

This repo is an independent reimplementation of the general approach
described in that report's public summary; no original report text, tables,
or charts are reproduced here, and none of the implementation choices below
should be attributed to the report's authors.

## Disclaimer

Research code for educational/personal use. Not investment advice; no warranty on data correctness or backtest performance.
