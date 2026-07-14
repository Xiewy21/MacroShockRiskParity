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

All market and macro data is fetched on first run via [akshare](https://github.com/akfamily/akshare) (free, public sources) and cached locally under `scripts/cache/` — no data files are bundled in this repo. Outputs (performance table + charts) land in `scripts/output/`.

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

## Disclaimer

Research code for educational/personal use. Not investment advice; no warranty on data correctness or backtest performance.
