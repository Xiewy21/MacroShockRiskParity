"""回测引擎：月末算权重、次月首个交易日收盘价调仓（买入持有至下次调仓，
组内权重随资产收益漂移）；对 Asset RP / Macro RP / Joint RP 三个策略在
config.CAP_LEVELS 每个 cap 档位上分别回测，输出绩效表与图表。

严格无前视：t 月末只使用截至 t（含）交易日的收益/冲击历史计算权重，
交易发生在 t+1 个交易日的收盘价，不会用到该笔交易之后才产生的信息。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import config
import data
import macro_factors
import shocks
import risk_parity as rp


def month_end_dates(calendar):
    s = calendar.to_series()
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).max().values)


def next_trading_day(date, calendar):
    pos = calendar.searchsorted(date, side="right")
    if pos >= len(calendar):
        return None
    return calendar[pos]


def rolling_window_returns(returns, calendar, end_date, window_days=config.ROLLING_WINDOW_DAYS):
    """截至 end_date（含）的最近 window_days 个交易日收益，仅保留窗口内无缺失的资产列（动态纳入）。"""
    end_pos = calendar.get_loc(end_date)
    start_pos = max(0, end_pos - window_days + 1)
    window_dates = calendar[start_pos:end_pos + 1]
    full = returns.loc[window_dates]
    avail_cols = full.columns[full.notna().all()]
    return full[avail_cols]


def compute_raw_weights(strategy, returns_window, shock_window, prev_joint_weight):
    if strategy == "Asset RP":
        return rp.asset_rp_weights(returns_window)
    if strategy == "Macro RP":
        return rp.macro_rp_weights(returns_window, shock_window)
    if strategy == "Joint RP":
        return rp.joint_rp_weights(returns_window, shock_window, prev_joint_weight)
    raise ValueError(strategy)


def drift_weight_vector(weight, asset_returns_row, target_universe):
    """按当日各资产收益率漂移持仓权重（买入持有），并扩展到 target_universe（新增资产权重记0）。"""
    w = weight.reindex(target_universe).fillna(0.0)
    r = asset_returns_row.reindex(target_universe).fillna(0.0)
    grown = w * (1.0 + r)
    total = grown.sum()
    if total <= 0:
        return w
    return grown / total


def run_single(strategy, cap, prices, returns, calendar, shock_df, rebalance_month_ends):
    """跑单个 (strategy, cap) 组合的完整回测，返回 daily nav Series、daily turnover Series、
    weight_history dict[date->Series]（仅在调仓日记录目标权重，用于持仓堆叠图）。
    """
    exec_dates = []
    target_weights_by_exec = {}
    prev_joint_weight = None

    for t in rebalance_month_ends:
        exec_date = next_trading_day(t, calendar)
        if exec_date is None:
            continue
        window = rolling_window_returns(returns, calendar, t)
        if window.shape[1] < 2 or window.shape[0] < config.ROLLING_WINDOW_DAYS:
            continue
        shock_window = shock_df.loc[window.index]

        w_opt = compute_raw_weights(strategy, window, shock_window, prev_joint_weight)
        vol_est = rp.mad_volatility(window)
        low_vol_names = rp.low_vol_group(vol_est)
        w_final = rp.apply_low_vol_cap(w_opt, low_vol_names, cap)

        if strategy == "Joint RP":
            prev_joint_weight = w_final

        exec_dates.append(exec_date)
        target_weights_by_exec[exec_date] = w_final

    if not exec_dates:
        return None

    first_exec = exec_dates[0]
    sim_dates = calendar[calendar >= first_exec]

    nav = 1.0
    nav_series = pd.Series(index=sim_dates, dtype=float)
    turnover_series = pd.Series(0.0, index=sim_dates)
    current_w = pd.Series(dtype=float)
    exec_set = set(exec_dates)

    for d in sim_dates:
        ret_row = returns.loc[d]
        if d in exec_set:
            target_w = target_weights_by_exec[d]
            universe = current_w.index.union(target_w.index)
            drifted = drift_weight_vector(current_w, ret_row, universe) if len(current_w) else pd.Series(0.0, index=universe)
            r_today = float((current_w.reindex(universe).fillna(0.0) * ret_row.reindex(universe).fillna(0.0)).sum()) if len(current_w) else 0.0

            target_full = target_w.reindex(universe).fillna(0.0)
            turnover = float((target_full - drifted).abs().sum())
            cost = turnover * config.TRANSACTION_COST_RATE
            nav *= (1.0 + r_today) * (1.0 - cost)
            turnover_series.loc[d] = turnover
            current_w = target_w
        else:
            r_today = float((current_w.reindex(ret_row.index).fillna(0.0) * ret_row.fillna(0.0)).sum()) if len(current_w) else 0.0
            nav *= (1.0 + r_today)
            current_w = drift_weight_vector(current_w, ret_row, current_w.index) if len(current_w) else current_w

        nav_series.loc[d] = nav

    return {
        "nav": nav_series,
        "turnover": turnover_series,
        "weights": target_weights_by_exec,
    }


def performance_table(nav_series, turnover_series):
    daily_ret = nav_series.pct_change().dropna()
    n_years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    ann_return = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    ann_vol = daily_ret.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = (ann_return - config.RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    cum = nav_series / nav_series.cummax()
    max_dd = (cum - 1.0).min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    avg_daily_turnover = turnover_series.reindex(nav_series.index).fillna(0.0).mean()

    yearly_returns = {}
    for year, grp in nav_series.groupby(nav_series.index.year):
        if len(grp) < 2:
            continue
        yearly_returns[year] = grp.iloc[-1] / grp.iloc[0] - 1.0

    return {
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "avg_daily_turnover": avg_daily_turnover,
        "yearly_returns": yearly_returns,
    }


def run_backtest():
    """跑全部 策略 x cap 档位组合，返回 results[(strategy, cap)] = run_single 输出。"""
    prices = data.get_asset_prices()
    returns = prices.pct_change()
    calendar = data.get_trading_calendar()

    group_signals = macro_factors.build_macro_group_signals(calendar)
    shock_df = shocks.build_shock_matrix(group_signals)

    # 月末决策日只需满足"次月首个交易日 >= 回测起点"即可，天然包含 2015-12-31
    # （决策用 2013-2015 三年滚动历史，执行落在 2016 年首个交易日，与"滚动数据
    # 从2013起、回测区间2016-01至今"的要求一致，无需再额外前置补一个月末）。
    all_month_ends = month_end_dates(calendar)
    backtest_start = pd.Timestamp(config.BACKTEST_START)
    rebalance_month_ends = [t for t in all_month_ends if next_trading_day(t, calendar) is not None
                             and next_trading_day(t, calendar) >= backtest_start]

    results = {}
    for strategy in config.STRATEGIES:
        for cap in config.CAP_LEVELS:
            out = run_single(strategy, cap, prices, returns, calendar, shock_df, rebalance_month_ends)
            if out is None:
                continue
            out["performance"] = performance_table(out["nav"], out["turnover"])
            results[(strategy, cap)] = out

    return results, prices


# ---------------------------------------------------------------------------
# 输出：绩效表 + 图表
# ---------------------------------------------------------------------------
def build_performance_dataframe(results):
    rows = []
    for (strategy, cap), out in results.items():
        perf = out["performance"]
        row = {
            "策略": strategy,
            "cap": cap,
            "年化收益": perf["annual_return"],
            "年化波动": perf["annual_vol"],
            "Sharpe": perf["sharpe"],
            "最大回撤": perf["max_drawdown"],
            "Calmar": perf["calmar"],
            "日均换手": perf["avg_daily_turnover"],
        }
        for year, r in sorted(perf["yearly_returns"].items()):
            row[f"{year}年收益"] = r
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["策略", "cap"]).reset_index(drop=True)
    return df


def plot_nav_comparison(results, cap, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in config.STRATEGIES:
        out = results.get((strategy, cap))
        if out is None:
            continue
        out["nav"].plot(ax=ax, label=strategy)
    ax.set_title(f"净值对比 (cap={cap})")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_yearly_excess(results, cap, out_path, base_strategy="Asset RP", target_strategy="Joint RP"):
    base = results.get((base_strategy, cap))
    target = results.get((target_strategy, cap))
    if base is None or target is None:
        return
    base_perf = base["performance"]["yearly_returns"]
    target_perf = target["performance"]["yearly_returns"]
    years = sorted(set(base_perf) & set(target_perf))
    excess = [target_perf[y] - base_perf[y] for y in years]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in excess]
    ax.bar([str(y) for y in years], excess, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{target_strategy} 相对 {base_strategy} 分年超额收益 (cap={cap})")
    ax.set_ylabel("超额收益")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_joint_holdings(results, cap, out_path):
    out = results.get(("Joint RP", cap))
    if out is None:
        return
    weights = out["weights"]
    df = pd.DataFrame(weights).T.sort_index().fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(df.index, df.values.T, labels=df.columns)
    ax.set_title(f"Joint RP 持仓权重堆叠图 (cap={cap})")
    ax.set_ylabel("权重")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
