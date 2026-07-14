"""Self-check script: no-look-ahead, optimizer analytic solution, cache health.

Run after any code change, before trusting backtest output:
    python scripts/validate.py
Exits 0 if every check that actually ran passes, 1 otherwise.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

import config
import risk_parity as rp
import shocks


class CheckFailed(Exception):
    pass


def check_no_lookahead():
    """A shock computed at time <= t from the full sample must match the shock
    computed at time <= t from a sample truncated at t: shocks.py must only use
    the Kalman *filtered* estimate (causal), never a *smoothed* one (which would
    use future observations and silently leak information into the backtest).
    """
    rng = np.random.default_rng(0)
    n = 480
    dates = pd.bdate_range("2015-01-01", periods=n)
    level = 0.0
    values = np.empty(n)
    for i in range(n):
        if i % 15 == 0:
            level += rng.normal(0.0, 1.0)
        values[i] = level
    series = pd.Series(values, index=dates)

    full_shock = shocks.extract_shock(series)

    cutoff = dates[n * 2 // 3]
    truncated_shock = shocks.extract_shock(series.loc[:cutoff])

    common = full_shock.loc[:cutoff].index.intersection(truncated_shock.index)
    a = full_shock.loc[common].values
    b = truncated_shock.loc[common].values
    if not np.allclose(a, b, atol=1e-8):
        max_diff = float(np.max(np.abs(a - b)))
        raise CheckFailed(
            f"shock series at/before cutoff {cutoff.date()} changed when future "
            f"data was appended (max diff {max_diff:.3e}) -- shocks.py appears to "
            f"be using smoothed rather than filtered estimates, which leaks "
            f"look-ahead information into the backtest."
        )


def check_rp_analytic_solution():
    """For 2 assets, the risk-parity solution is w_i ∝ 1/sigma_i for *any*
    valid correlation (the cross term cancels in the equal-risk-contribution
    condition) -- a closed form we can check the iterative solver against.
    """
    sigma = np.array([0.15, 0.25])
    rho = 0.4
    cov = np.array([
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ])
    w = rp.risk_parity_weights(cov)
    analytic = (1.0 / sigma) / (1.0 / sigma).sum()
    max_err = float(np.max(np.abs(w - analytic)))
    if max_err >= 1e-6:
        raise CheckFailed(
            f"iterative risk-parity solver diverges from the closed-form "
            f"solution (w_i ∝ 1/sigma_i): got {w}, expected {analytic}, "
            f"max error {max_err:.3e} (tolerance 1e-6)"
        )


# Documented dynamic-universe exceptions (see config.py ASSET_UNIVERSE comment):
# these two sources genuinely can't provide history back to config.DATA_START.
_SHORT_HISTORY_ASSET_START = {
    "黄金AU9999": "2016-12-19",
    "南华商品指数": "2022-07-01",
}


def _estimate_missing_rate(idx):
    """Frequency-agnostic gap check: infer daily/weekly/monthly cadence from the
    median gap between observations, then compare actual count to the count
    that cadence implies over the observed span.
    """
    if len(idx) < 3:
        return 0.0
    gaps = idx.to_series().diff().dropna().dt.days.values
    median_gap = float(np.median(gaps))
    span_days = (idx[-1] - idx[0]).days
    if span_days <= 0:
        return 0.0
    if median_gap <= 3:
        expected = span_days * 5.0 / 7.0  # daily / business-day cadence
    elif median_gap <= 10:
        expected = span_days / 7.0  # weekly
    else:
        expected = span_days / 30.44  # monthly or coarser
    if expected <= 0:
        return 0.0
    return max(0.0, 1.0 - len(idx) / expected)


def check_cache_health():
    """Sanity-check whatever is already in cache/: does it start near
    config.DATA_START (or a documented dynamic-universe exception), is it
    reasonably fresh (<=90 days old), and is the observation density plausible
    for its apparent frequency. Skips entirely if cache/ is empty -- this is a
    healthcheck for existing data, not a trigger to fetch any.
    """
    paths = sorted(glob.glob(os.path.join(config.CACHE_DIR, "*.csv")))
    if not paths:
        print("  [skip] cache/ is empty -- nothing to check (run main.py once to populate it)")
        return

    data_start = pd.Timestamp(config.DATA_START)
    recency_floor = pd.Timestamp.today() - pd.Timedelta(days=90)
    problems = []

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            problems.append(f"{name}: cache file is empty")
            continue
        idx = df.index.sort_values()

        asset_key = name[len("asset_"):] if name.startswith("asset_") else None
        expected_start = pd.Timestamp(_SHORT_HISTORY_ASSET_START.get(asset_key, data_start))

        if idx[0] > expected_start + pd.Timedelta(days=400):
            problems.append(
                f"{name}: earliest cached date {idx[0].date()} is far later than "
                f"expected start {expected_start.date()}"
            )
        if idx[-1] < recency_floor:
            problems.append(f"{name}: latest cached date {idx[-1].date()} is more than 90 days old")

        missing_rate = _estimate_missing_rate(idx)
        if missing_rate >= 0.05:
            problems.append(f"{name}: estimated missing rate {missing_rate:.1%} >= 5%")

    if problems:
        raise CheckFailed("cache health issues found:\n  - " + "\n  - ".join(problems))


def main():
    checks = [
        ("no-look-ahead (shocks.py filtered vs smoothed)", check_no_lookahead),
        ("risk-parity solver vs analytic solution", check_rp_analytic_solution),
        ("cache health", check_cache_health),
    ]
    failed = False
    for label, fn in checks:
        print(f"[check] {label}")
        try:
            fn()
        except CheckFailed as exc:
            failed = True
            print(f"  [FAIL] {exc}")
        else:
            print("  [ok]")

    if failed:
        print("\nvalidate.py: FAILED")
        sys.exit(1)
    print("\nvalidate.py: all checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
