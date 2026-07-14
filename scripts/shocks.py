"""预期外冲击提取：局部线性趋势状态空间模型 + 在线 Kalman 递推。

状态 x_t = [level_t, trend_t]'，转移 T = [[1,1],[0,1]]（趋势为随机游走）。
每步预测前对状态协方差乘折现矩阵（半衰期3个月，按经过的交易日数做幂次
折现），实现"越久远的历史记忆权重越低"的自适应学习；状态扰动强度 Q 用
过去1年样本的一阶/二阶差分方差估计。

关键设计：宏观合成信号在两次数据发布之间是分段常数（前向填充），若逐日
对这种阶梯序列跑 Kalman 滤波，差分方差会被大量的"零变化"日拉得极小，
一旦真正发布新数据就会产生数值上爆炸的标准化新息（远超合理量级）。因此
滤波只在信号真正发生变化的"发布日"上递推（用发布间隔的交易日数 dt 做
折现的指数），非发布日的冲击记为 0（无新信息=无冲击），再重新对齐到完整
交易日历，与日度收益率对齐用于回归。这样冲击的量级在跨维度之间才具有
可比性，且严禁使用未来信息（每次递推只用当期及之前的发布）。
"""

import numpy as np
import pandas as pd

import config

_T = np.array([[1.0, 1.0], [0.0, 1.0]])
_Z = np.array([1.0, 0.0])
_DECAY = 0.5 ** (1.0 / config.SHOCK_HALFLIFE_DAYS)


def _estimate_qr(window_vals):
    """用过去1年"实际发布事件"样本的一阶/二阶差分方差，估计 (q_level, q_trend, r_obs)。"""
    d1 = np.diff(window_vals)
    if len(d1) < 2:
        return 1e-4, 1e-4, 1e-4
    d2 = np.diff(d1)
    var_d1 = np.nanvar(d1) if len(d1) else 1e-4
    var_d2 = np.nanvar(d2) if len(d2) else 1e-4
    q_trend = max(var_d2, 1e-8)
    q_level = max(var_d1 - 2 * q_trend, 1e-8)
    r_obs = max(0.1 * var_d1, 1e-8)
    return q_level, q_trend, r_obs


def _event_dates(series):
    """提取信号真正发生变化的日期（含首个有效值），即"发布日"序列。"""
    valid = series.dropna()
    if valid.empty:
        return valid
    changed = valid.diff().fillna(1.0) != 0.0
    return valid[changed]


def extract_shock(series):
    """对单个宏观维度日度合成信号做 Kalman 滤波，返回标准化 innovation 序列
    （非发布日记为0，与原 series 的交易日历完全对齐）。
    """
    events = _event_dates(series)
    daily_shock = pd.Series(0.0, index=series.index)
    if len(events) < 2:
        return daily_shock

    calendar = series.index
    event_positions = [calendar.get_loc(d) for d in events.index]
    values = events.values

    lookback = config.SHOCK_VAR_LOOKBACK_DAYS
    min_periods = config.SHOCK_VAR_MIN_PERIODS

    m = np.array([values[0], 0.0])
    P = np.eye(2) * 1.0

    for i in range(1, len(values)):
        dt = event_positions[i] - event_positions[i - 1]
        dt = max(dt, 1)

        window_start_pos = event_positions[i] - lookback
        window_vals = [values[j] for j in range(i) if event_positions[j] >= window_start_pos]
        window_vals.append(values[i])
        if len(window_vals) >= min_periods:
            q_level, q_trend, r_obs = _estimate_qr(np.array(window_vals))
        else:
            # 历史事件不足1年样本量时的"扩散先验"：用当前已有全部历史的方差
            # 做量级近似，而非任意给一个极小的默认值（避免早期样本太少时
            # F 过小导致标准化新息数值爆炸）。
            prior_var = max(np.nanvar(values[: i + 1]), 1e-4)
            q_level, q_trend, r_obs = prior_var, prior_var * 0.1, prior_var
        Q = np.diag([q_level, q_trend])

        decay_dt = _DECAY ** dt
        discount = np.diag([decay_dt, decay_dt])

        # 1) 折现（按经过的交易日数做幂次折现）
        P_disc = discount @ P @ discount.T
        # 2) 预测（趋势项按 dt 步递推）
        T_dt = np.array([[1.0, dt], [0.0, 1.0]])
        m_pred = T_dt @ m
        P_pred = T_dt @ P_disc @ T_dt.T + Q
        # 3) 新息（一步前瞻预测误差）与其标准差
        y_t = values[i]
        v = y_t - _Z @ m_pred
        F = _Z @ P_pred @ _Z.T + r_obs
        F = max(F, 1e-12)
        shock_val = v / np.sqrt(F)
        daily_shock.iloc[event_positions[i]] = shock_val
        # 4) 更新
        K = P_pred @ _Z.T / F
        m = m_pred + K * v
        P = P_pred - np.outer(K, _Z @ P_pred)

    return daily_shock


def build_shock_matrix(macro_group_signals):
    """对六维宏观合成信号逐列提取标准化冲击，返回 DataFrame(index, columns=六维)。"""
    shocks = {}
    for col in macro_group_signals.columns:
        shocks[col] = extract_shock(macro_group_signals[col])
    return pd.DataFrame(shocks, index=macro_group_signals.index)
