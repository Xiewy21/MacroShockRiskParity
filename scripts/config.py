"""全局配置：资产池、宏观指标分组、滚动窗口与模型超参数。

所有可调参数集中于此，其余模块只读取本文件，不重复定义参数。
"""

import os

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 时间区间
# ---------------------------------------------------------------------------
DATA_START = "2013-01-01"      # 数据抓取起点（为滚动窗口预留历史）
BACKTEST_START = "2016-01-01"  # 回测起点
DATA_END = None                # None = 抓取到今天

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# 资产池：名称 -> akshare 抓取规格
#   source 字段决定 data.py 中使用哪个抓取函数
#   currency 标注原始计价货币，非 CNY 的资产会被转换为人民币口径
# 注：南华商品指数原生接口已在 akshare 中下架，用中证商品期货指数（2022-07 起）
#     替代；黄金 AU9999 仅能取到 2016-12-19 之后的数据。两者均采用"动态纳入"
#     处理：某资产滚动窗口历史不足 ROLLING_WINDOW_DAYS 时，该资产当期权重强制为 0，
#     其余资产回测不受影响，详见 risk_parity.py 的 dynamic universe 处理。
# ---------------------------------------------------------------------------
ASSET_UNIVERSE = {
    "沪深300全收益": {"source": "csindex", "symbol": "H00300", "currency": "CNY"},
    "中证500全收益": {"source": "csindex", "symbol": "H00905", "currency": "CNY"},
    "恒生指数": {"source": "hsi", "symbol": "HSI", "currency": "HKD"},
    "标普500": {"source": "sp500", "symbol": ".INX", "currency": "USD"},
    "中债国债总财富指数": {"source": "cbond_wealth", "index_category": "国债总指数", "currency": "CNY"},
    "中债信用债总财富指数": {"source": "cbond_wealth", "index_category": "信用债总指数", "currency": "CNY"},
    "中证转债": {"source": "csindex", "symbol": "000832", "currency": "CNY"},
    "黄金AU9999": {"source": "gold_sge", "symbol": "Au99.99", "currency": "CNY"},
    "南华商品指数": {"source": "commodity_proxy", "symbol": "中证商品期货指数", "currency": "CNY"},
}

# ---------------------------------------------------------------------------
# 宏观指标六维分组
#   freq: 'monthly' 或 'weekly'，决定发布滞后规则与重采样频率
#   lag: 滞后期数（月度滞后1月，周度滞后1周）
# ---------------------------------------------------------------------------
MACRO_GROUPS = {
    "增长": [
        {"name": "PMI", "source": "pmi", "freq": "monthly", "lag": 1},
        {"name": "工业增加值", "source": "iva", "freq": "monthly", "lag": 1},
    ],
    "通胀": [
        {"name": "CPI", "source": "cpi", "freq": "monthly", "lag": 1},
        {"name": "PPI", "source": "ppi", "freq": "monthly", "lag": 1},
    ],
    "流动性": [
        {"name": "M2", "source": "m2", "freq": "monthly", "lag": 1},
        {"name": "社融", "source": "shrzgm", "freq": "monthly", "lag": 1},
        {"name": "SHIBOR", "source": "shibor", "freq": "weekly", "lag": 1},
    ],
    "利率": [
        {"name": "10Y国债收益率", "source": "cn10y", "freq": "weekly", "lag": 1},
    ],
    "信用": [
        {"name": "信用利差", "source": "credit_spread", "freq": "weekly", "lag": 1},
    ],
    "外部": [
        {"name": "美元指数", "source": "dxy", "freq": "weekly", "lag": 1},
        {"name": "美债10Y", "source": "us10y", "freq": "weekly", "lag": 1},
    ],
}
MACRO_GROUP_ORDER = ["增长", "通胀", "流动性", "利率", "信用", "外部"]
N_MACRO_FACTORS = len(MACRO_GROUP_ORDER)

# ---------------------------------------------------------------------------
# 宏观因子处理参数
# ---------------------------------------------------------------------------
ZSCORE_WINDOW_YEARS = 3
ZSCORE_MIN_PERIODS_MONTHS = 12   # 至少 12 个月观测值才开始输出 z-score

# ---------------------------------------------------------------------------
# 冲击提取（局部线性趋势 Kalman 滤波）参数
# ---------------------------------------------------------------------------
SHOCK_HALFLIFE_DAYS = 63         # 3 个月半衰期（按交易日近似 3*21）
SHOCK_VAR_LOOKBACK_DAYS = 252    # 状态扰动强度估计窗口（过去1年）
SHOCK_VAR_MIN_PERIODS = 60       # 方差估计所需最少观测

# ---------------------------------------------------------------------------
# 风险平价参数
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS = ZSCORE_WINDOW_YEARS * TRADING_DAYS_PER_YEAR  # 3年滚动窗口（协方差/回归共用）
SHRINKAGE_INTENSITY = 0.3        # 宏观因子协方差压缩强度（目标=对角阵）
JOINT_MIX_WEIGHT = 0.5           # Joint RP 中样本协方差与宏观隐含协方差各自权重
SMOOTHING_LAMBDA = 0.2           # w_t = LAMBDA * w_opt + (1-LAMBDA) * w_{t-1}，仅用于 Joint RP

RP_TOLERANCE = 1e-8
RP_MAX_ITER = 200
RP_LEARNING_RATE = 0.5           # 风险贡献偏差比例调权的学习率指数（lr=1 在部分协方差结构下会产生周期2振荡不收敛，0.5经验证稳健收敛）
RP_RATIO_CLIP = (0.5, 2.0)       # 单步调权比例上下限，防止资产间存在较强负相关时风险贡献跨越正负号导致的振荡不收敛

LOW_VOL_GROUP_SIZE = 5
CAP_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_CAP_FOR_PLOT = 0.4       # 持仓堆叠图使用的默认 cap 档位

# ---------------------------------------------------------------------------
# 回测参数
# ---------------------------------------------------------------------------
TRANSACTION_COST_RATE = 0.0005   # 双边费率万5（每单位换手成本）
RISK_FREE_RATE = 0.02            # 年化无风险利率（近似）

STRATEGIES = ["Asset RP", "Macro RP", "Joint RP"]
