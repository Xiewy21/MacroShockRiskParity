"""宏观因子处理：当期可得对齐 -> 3年滚动z-score -> 组内正交化 -> 等权合成。

严格遵循"当期可得"原则：所有滚动统计量在 t 时点只使用 t（含）之前已发布
的信息；月度指标滞后1月、周度指标滞后1周才视为可得，再映射到下一个最近
交易日并前向填充，与日度收益对齐。
"""

import numpy as np
import pandas as pd

import config
import data


def _release_lag(raw, freq, lag):
    """把指标的"统计所属期"索引平移为"市场可获知"日期。"""
    if freq == "monthly":
        offset = pd.DateOffset(months=lag)
    elif freq == "weekly":
        offset = pd.DateOffset(weeks=lag)
    else:
        raise ValueError(f"unknown freq {freq}")
    out = raw.copy()
    out.index = out.index + offset
    return out.sort_index()


def rolling_zscore(series, window_years=config.ZSCORE_WINDOW_YEARS, min_periods=None):
    """按时间窗口（而非按行数）做滚动 z-score，避免混用不同频率时窗口长度失真。"""
    if min_periods is None:
        min_periods = config.ZSCORE_MIN_PERIODS_MONTHS
    window = f"{int(window_years * 365.25)}D"
    roll = series.rolling(window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std()
    z = (series - mean) / std.replace(0, np.nan)
    return z


def map_to_trading_days(series_lagged, calendar):
    """把（已做发布滞后处理的）低频序列映射到下一个最近交易日并前向填充。"""
    combined_index = series_lagged.index.union(calendar)
    filled = series_lagged.reindex(combined_index).sort_index().ffill()
    return filled.reindex(calendar)


def _min_periods_for_freq(freq):
    # 月度：至少12个观测（1年）；周度：按约4.345周/月折算
    months = config.ZSCORE_MIN_PERIODS_MONTHS
    if freq == "monthly":
        return months
    return int(months * 4.345)


def build_indicator_daily_zscore(spec, calendar):
    """单个宏观指标：抓取原始值 -> 滞后 -> 原生频率滚动z-score -> 映射到交易日。"""
    raw = data.get_macro_raw(spec["source"])
    lagged = _release_lag(raw, spec["freq"], spec["lag"])
    min_periods = _min_periods_for_freq(spec["freq"])
    z_native = rolling_zscore(lagged, min_periods=min_periods)
    z_daily = map_to_trading_days(z_native, calendar)
    return z_daily


def _rolling_gs_orthogonalize(z_frame):
    """组内 Gram-Schmidt 正交化（顺序 = DataFrame 列顺序）。

    每一步用滚动窗口的历史（含当期）协方差/方差比估计投影系数，再从当期原始
    z 值中减去该投影，得到与此前正交基不相关的残差；最后对残差重新做一次
    滚动 z-score 使其量纲与其余分量一致，便于等权平均。

    注意：z_frame 是已经映射到每日交易日历、前向填充的阶梯序列。若直接在
    整个日历上做时间窗滚动协方差，窗口边界会逐日移动（即使没有新数据发布，
    最老的样本每天都在滑出窗口），导致 beta 在"数值本应完全不变"的平坦期内
    产生微小漂移，进而在下游 Kalman 冲击提取里污染方差估计。因此这里先抽取
    "事件行"（任一指标相对上一行发生变化的日期，含首行），只在事件行上滚动
    估计 beta，再前向填充回完整日历——这样平坦期内 beta 与残差都严格不变。
    """
    window = f"{int(config.ZSCORE_WINDOW_YEARS * 365.25)}D"
    min_periods = _min_periods_for_freq("monthly")
    cols = list(z_frame.columns)

    changed = z_frame.diff().abs().gt(0).any(axis=1)
    changed.iloc[0] = True
    event_frame = z_frame[changed]

    ortho_events = pd.DataFrame(index=event_frame.index)
    ortho_events[cols[0]] = event_frame[cols[0]]

    for i in range(1, len(cols)):
        resid = event_frame[cols[i]].copy()
        for j in range(i):
            basis = ortho_events[cols[j]]
            paired = pd.concat([event_frame[cols[i]], basis], axis=1).dropna()
            cov = paired.iloc[:, 0].rolling(window, min_periods=min_periods).cov(paired.iloc[:, 1])
            var = basis.rolling(window, min_periods=min_periods).var()
            beta = (cov / var.replace(0, np.nan)).reindex(event_frame.index)
            resid = resid - beta * basis
        ortho_events[cols[i]] = resid

    standardized_events = ortho_events.apply(lambda s: rolling_zscore(s, min_periods=min_periods))
    standardized = standardized_events.reindex(z_frame.index.union(standardized_events.index)) \
        .sort_index().ffill().reindex(z_frame.index)
    return standardized


def build_macro_group_signals(calendar):
    """返回 DataFrame(index=calendar, columns=六维分组)，逐日、仅用历史信息的合成信号。"""
    group_signals = {}
    for group, specs in config.MACRO_GROUPS.items():
        indicator_z = {}
        for spec in specs:
            indicator_z[spec["name"]] = build_indicator_daily_zscore(spec, calendar)
        z_frame = pd.DataFrame(indicator_z, index=calendar)

        if z_frame.shape[1] == 1:
            composite = z_frame.iloc[:, 0]
        else:
            ortho = _rolling_gs_orthogonalize(z_frame)
            composite = ortho.mean(axis=1)

        group_signals[group] = composite

    return pd.DataFrame(group_signals, index=calendar)[config.MACRO_GROUP_ORDER]
