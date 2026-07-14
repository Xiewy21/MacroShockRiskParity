"""风险平价核心：Asset RP / Macro RP / Joint RP 权重求解 + 低波资产上限约束。

所有函数只接受调用方已经切好的"历史窗口"数据（严格 t 期之前/含 t 的信息），
本模块不做任何时间对齐或前视相关的判断，保证无前视完全由 backtest.py 的滚动
窗口切片来保证。资产的"动态纳入"（历史不足则不参与当期优化）体现为：
调用方只传入当期已有完整窗口历史的资产列（NaN 列应在传入前 dropna）。
"""

import numpy as np
import pandas as pd

import config


def risk_parity_weights(cov, target=None, lr=config.RP_LEARNING_RATE,
                         tol=config.RP_TOLERANCE, max_iter=config.RP_MAX_ITER):
    """风险贡献偏差迭代法：等权目标（或自定义 target），仅多头，归一化。

    cov: n x n 协方差矩阵（numpy array）
    返回: 长度为 n 的 numpy 权重向量，和为 1。

    注：资产间存在较强负相关时（如债券对股票），某资产的边际风险贡献可能
    在权重迭代过程中跨越正负号，若不加约束，"target/rc" 比值会瞬间发散，
    导致权重在两个极端角点间周期振荡、永不收敛。这里对单步调权比例做
    上下限截断（RP_RATIO_CLIP），不改变收敛到等风险贡献不动点的性质
    （不动点处比例恒为1，截断不影响），但能有效抑制振荡。
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    if target is None:
        target = np.full(n, 1.0 / n)

    clip_lo, clip_hi = config.RP_RATIO_CLIP
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        port_var = w @ cov @ w
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = w * mrc / port_var
        if np.max(np.abs(rc - target)) < tol:
            break
        rc_safe = np.where(rc > 1e-10, rc, 1e-10)
        ratio = np.clip(target / rc_safe, clip_lo, clip_hi)
        w = w * ratio ** lr
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = w / s if s > 0 else np.full(n, 1.0 / n)
    return w


def sample_covariance(returns_window):
    """returns_window: DataFrame[t, asset]，逐日收益，返回 DataFrame 协方差矩阵。"""
    return returns_window.cov()


def asset_rp_weights(returns_window):
    """Asset RP：资产日收益样本协方差 -> 风险平价权重。"""
    cov = sample_covariance(returns_window)
    w = risk_parity_weights(cov.values)
    return pd.Series(w, index=cov.index)


def _rolling_regression_exposure(returns_window, shocks_window):
    """3年滚动回归：资产收益对6维冲击的暴露矩阵 B，及残差方差、因子协方差。

    返回 (B: DataFrame[asset, factor], resid_var: Series[asset], Sigma_f: DataFrame[factor, factor])
    """
    aligned = returns_window.join(shocks_window, how="inner", lsuffix="_r")
    assets = list(returns_window.columns)
    factors = list(shocks_window.columns)
    R = aligned[assets].values
    F = aligned[factors].values
    T = F.shape[0]
    Fc = np.column_stack([np.ones(T), F])
    coefs, _, _, _ = np.linalg.lstsq(Fc, R, rcond=None)
    B = coefs[1:, :].T  # n_assets x n_factors
    fitted = Fc @ coefs
    resid = R - fitted
    resid_var = resid.var(axis=0, ddof=1)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)

    B_df = pd.DataFrame(B, index=assets, columns=factors)
    resid_series = pd.Series(resid_var, index=assets)
    Sigma_f_df = pd.DataFrame(Sigma_f, index=factors, columns=factors)
    return B_df, resid_series, Sigma_f_df


def macro_implied_covariance(returns_window, shocks_window, shrinkage=config.SHRINKAGE_INTENSITY):
    """宏观隐含协方差 = B * Sigma_f(收缩后) * B' + diag(残差方差)。"""
    B, resid_var, Sigma_f = _rolling_regression_exposure(returns_window, shocks_window)
    target_diag = np.diag(np.diag(Sigma_f.values))
    Sigma_f_shrunk = (1 - shrinkage) * Sigma_f.values + shrinkage * target_diag
    Sigma_macro = B.values @ Sigma_f_shrunk @ B.values.T + np.diag(resid_var.values)
    return pd.DataFrame(Sigma_macro, index=B.index, columns=B.index)


def macro_rp_weights(returns_window, shocks_window):
    """Macro RP：宏观隐含协方差 -> 风险平价权重。"""
    cov_macro = macro_implied_covariance(returns_window, shocks_window)
    w = risk_parity_weights(cov_macro.values)
    return pd.Series(w, index=cov_macro.index)


def _normalize_by_trace(cov):
    tr = np.trace(cov.values)
    if tr <= 0:
        return cov
    return cov / tr


def joint_covariance(cov_asset, cov_macro, mix=config.JOINT_MIX_WEIGHT):
    """样本协方差与宏观隐含协方差各自按迹归一化后加权混合。"""
    a_norm = _normalize_by_trace(cov_asset)
    m_norm = _normalize_by_trace(cov_macro.reindex(index=cov_asset.index, columns=cov_asset.columns))
    return mix * a_norm + (1 - mix) * m_norm


def smooth_weights(w_opt, w_prev, lam=config.SMOOTHING_LAMBDA):
    """w_t = lam * w_opt + (1-lam) * w_prev（动态纳入：新资产在 w_prev 中记为0）。"""
    if w_prev is None or len(w_prev) == 0:
        return w_opt
    prev_aligned = w_prev.reindex(w_opt.index).fillna(0.0)
    w = lam * w_opt + (1 - lam) * prev_aligned
    s = w.sum()
    return w / s if s > 0 else w_opt


def joint_rp_weights(returns_window, shocks_window, prev_weights):
    """Joint RP：混合协方差 -> 风险平价 -> 权重平滑（仅 Joint RP 做平滑）。"""
    cov_asset = sample_covariance(returns_window)
    cov_macro = macro_implied_covariance(returns_window, shocks_window)
    cov_joint = joint_covariance(cov_asset, cov_macro)
    w_opt = pd.Series(risk_parity_weights(cov_joint.values), index=cov_joint.index)
    return smooth_weights(w_opt, prev_weights)


def mad_volatility(returns_window):
    """用 MAD（中位数绝对偏差，标定为与 std 可比的尺度）估计截面波动率。"""
    med = returns_window.median()
    mad = (returns_window - med).abs().median()
    return mad * 1.4826


def low_vol_group(vol_series, group_size=config.LOW_VOL_GROUP_SIZE):
    """按 MAD 波动率升序排列，取波动最小的 group_size 个资产名称集合。"""
    n = min(group_size, len(vol_series))
    return set(vol_series.sort_values().index[:n])


def apply_low_vol_cap(weights, low_vol_names, cap):
    """低波组总权重超过 cap 时按比例压缩，释放权重按非低波资产原始占比分配。"""
    names = [n for n in low_vol_names if n in weights.index]
    if not names:
        return weights
    low_sum = weights[names].sum()
    if low_sum <= cap or low_sum <= 0:
        return weights

    w = weights.copy()
    w[names] = weights[names] * (cap / low_sum)
    released = low_sum - cap

    other_names = [n for n in weights.index if n not in names]
    other_sum = weights[other_names].sum()
    if other_sum > 0:
        w[other_names] = weights[other_names] + released * (weights[other_names] / other_sum)
    return w
