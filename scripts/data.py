"""数据层：封装 akshare 行情与宏观数据抓取，所有结果本地 CSV 缓存，优先读缓存。

对外主要接口：
    get_asset_prices()   -> DataFrame[date, asset_name] 人民币口径日收盘价（未剔除节假日对齐）
    get_trading_calendar()-> DatetimeIndex 交易日历（以沪深300为基准）
    get_macro_raw(spec)  -> Series 单个宏观指标原始（未做滞后/标准化）时间序列
"""

import os
import time
import akshare as ak
import numpy as np
import pandas as pd

import config


def _cache_path(name):
    return os.path.join(config.CACHE_DIR, f"{name}.csv")


def _cached(name, fetch_fn):
    """本地 CSV 缓存装饰：存在则读取，否则抓取后写入。"""
    path = _cache_path(name)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    df = _retry(fetch_fn)
    df.to_csv(path)
    return df


def _retry(fn, attempts=4, backoff=2.0):
    """akshare 部分接口（东方财富/新浪）偶发连接中断，做简单重试。"""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            time.sleep(backoff * (i + 1))
    raise last_exc


# ---------------------------------------------------------------------------
# 资产价格抓取
# ---------------------------------------------------------------------------
def _fetch_csindex(symbol):
    df = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date="20130101", end_date="20301231")
    df = df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


def _fetch_hsi():
    df = ak.stock_hk_index_daily_sina(symbol="HSI")
    df = df[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


def _fetch_sp500():
    df = ak.index_us_stock_sina(symbol=".INX")
    df = df[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


def _fetch_cbond_wealth(index_category):
    df = ak.bond_index_general_cbond(index_category=index_category, indicator="财富", period="总值")
    df = df.rename(columns={"value": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


def _fetch_gold_sge():
    df = ak.spot_hist_sge(symbol="Au99.99")
    df = df[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


def _fetch_commodity_proxy(symbol):
    df = ak.futures_index_ccidx(symbol=symbol)
    df = df.rename(columns={"日期": "date", "收盘点位": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


def _fetch_forex(symbol):
    df = ak.forex_hist_em(symbol=symbol)
    df = df.rename(columns={"日期": "date", "最新价": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["close"]]


_ASSET_FETCHERS = {
    "csindex": lambda spec: _fetch_csindex(spec["symbol"]),
    "hsi": lambda spec: _fetch_hsi(),
    "sp500": lambda spec: _fetch_sp500(),
    "cbond_wealth": lambda spec: _fetch_cbond_wealth(spec["index_category"]),
    "gold_sge": lambda spec: _fetch_gold_sge(),
    "commodity_proxy": lambda spec: _fetch_commodity_proxy(spec["symbol"]),
}


def get_raw_asset_price(name):
    """返回单个资产原始计价货币下的收盘价 Series（未转换人民币）。"""
    spec = config.ASSET_UNIVERSE[name]
    fetcher = _ASSET_FETCHERS[spec["source"]]
    df = _cached(f"asset_{name}", lambda: fetcher(spec))
    return df["close"].rename(name)


def get_fx_series(symbol):
    """symbol: 'USDCNH' 或 'USDHKD'，返回 CNH/HKD 计价 Series。"""
    df = _cached(f"fx_{symbol}", lambda: _fetch_forex(symbol))
    return df["close"].rename(symbol)


def get_trading_calendar():
    """以沪深300全收益指数的交易日作为主日历（境内交易日历）。"""
    s = get_raw_asset_price("沪深300全收益")
    return s.index


def get_asset_prices():
    """返回所有资产人民币口径收盘价 DataFrame，reindex 到主交易日历并前向填充
    （境外资产在境内假期无新价，用前值近似当日估值，避免引入未来信息）。
    """
    calendar = get_trading_calendar()
    usdcnh = get_fx_series("USDCNH").reindex(calendar).ffill()
    usdhkd = get_fx_series("USDHKD").reindex(calendar).ffill()
    hkdcnh = usdcnh / usdhkd

    prices = {}
    for name, spec in config.ASSET_UNIVERSE.items():
        raw = get_raw_asset_price(name)
        raw = raw.reindex(raw.index.union(calendar)).sort_index().ffill().reindex(calendar)
        if spec["currency"] == "HKD":
            raw = raw * hkdcnh
        elif spec["currency"] == "USD":
            raw = raw * usdcnh
        prices[name] = raw
    return pd.DataFrame(prices, index=calendar)


# ---------------------------------------------------------------------------
# 宏观数据抓取
# ---------------------------------------------------------------------------
def _parse_year_month(s):
    """兼容 '2026年06月份' / '202606' / '2026.6' / 标准日期 等多种月份字符串格式。"""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y年%m月份", errors="coerce")
    mask = out.isna()
    if mask.any():
        out2 = pd.to_datetime(s[mask], format="%Y%m", errors="coerce")
        out[mask] = out2
        mask = out.isna()
    if mask.any():
        out3 = pd.to_datetime(s[mask], format="%Y.%m", errors="coerce")
        out[mask] = out3
        mask = out.isna()
    if mask.any():
        out[mask] = pd.to_datetime(s[mask], errors="coerce")
    return out


def _fetch_pmi():
    df = ak.macro_china_pmi()
    df = df.rename(columns={"月份": "date", "制造业-指数": "value"})[["date", "value"]]
    df["date"] = _parse_year_month(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _fetch_iva():
    df = ak.macro_china_gyzjz()
    df = df.rename(columns={"月份": "date", "同比增长": "value"})[["date", "value"]]
    df["date"] = _parse_year_month(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _fetch_cpi():
    df = ak.macro_china_cpi()
    df = df.rename(columns={"月份": "date", "全国-同比增长": "value"})[["date", "value"]]
    df["date"] = _parse_year_month(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _fetch_ppi():
    df = ak.macro_china_ppi()
    df = df.rename(columns={"月份": "date", "当月同比增长": "value"})[["date", "value"]]
    df["date"] = _parse_year_month(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _fetch_m2():
    df = ak.macro_china_supply_of_money()
    col = [c for c in df.columns if "M2" in c and "同比" in c][0]
    df = df.rename(columns={"统计时间": "date", col: "value"})[["date", "value"]]
    df["date"] = _parse_year_month(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _fetch_shrzgm():
    df = ak.macro_china_shrzgm()
    df = df.rename(columns={"月份": "date", "社会融资规模增量": "value"})[["date", "value"]]
    df["date"] = df["date"].astype(str)
    df["date"] = _parse_year_month(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")


def _fetch_shibor():
    df = ak.macro_china_shibor_all()
    col = [c for c in df.columns if c.startswith("3M")][0]
    df = df.rename(columns={"日期": "date", col: "value"})[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df.set_index("date")


def _fetch_bond_zh_us_rate():
    df = ak.bond_zh_us_rate(start_date="20130101")
    df["日期"] = pd.to_datetime(df["日期"])
    return df.set_index("日期").sort_index()


def _fetch_cn10y():
    raw = _cached("raw_bond_zh_us_rate", _fetch_bond_zh_us_rate)
    col = [c for c in raw.columns if "中国国债收益率10年" == c or ("中国国债收益率10年" in c and "-" not in c)]
    col = col[0] if col else "中国国债收益率10年"
    s = raw[col].rename("value").to_frame()
    return s


def _fetch_us10y():
    raw = _cached("raw_bond_zh_us_rate", _fetch_bond_zh_us_rate)
    col = [c for c in raw.columns if "美国国债收益率10年" == c or ("美国国债收益率10年" in c and "-" not in c)]
    col = col[0] if col else "美国国债收益率10年"
    s = raw[col].rename("value").to_frame()
    return s


def _fetch_bond_china_yield_all():
    frames = []
    start_year = 2013
    end_year = pd.Timestamp.today().year
    for year in range(start_year, end_year + 1):
        try:
            df = ak.bond_china_yield(start_date=f"{year}0101", end_date=f"{year}1231")
        except Exception:
            continue
        if df is not None and len(df):
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["日期"] = pd.to_datetime(out["日期"])
    return out


def _fetch_credit_spread():
    raw = _cached("raw_bond_china_yield", _fetch_bond_china_yield_all)
    govt = raw[raw["曲线名称"] == "中债国债收益率曲线"][["日期", "10年"]].rename(columns={"10年": "govt"})
    credit = raw[raw["曲线名称"] == "中债中短期票据收益率曲线(AAA)"][["日期", "10年"]].rename(columns={"10年": "credit"})
    merged = pd.merge(govt, credit, on="日期", how="inner").sort_values("日期")
    merged["value"] = merged["credit"] - merged["govt"]
    return merged.set_index("日期")[["value"]]


_DXY_WEIGHTS = {
    "EURUSD": -0.576,
    "USDJPY": 0.136,
    "GBPUSD": -0.119,
    "USDCAD": 0.091,
    "USDSEK": 0.042,
    "USDCHF": 0.036,
}


def _fetch_dxy():
    series = {}
    for sym in _DXY_WEIGHTS:
        s = get_fx_series(sym)
        series[sym] = s
    df = pd.DataFrame(series).sort_index().ffill().dropna()
    log_level = sum(_DXY_WEIGHTS[sym] * np.log(df[sym]) for sym in _DXY_WEIGHTS)
    value = 50.14348112 * np.exp(log_level)
    out = value.rename("value").to_frame()
    return out


_MACRO_FETCHERS = {
    "pmi": _fetch_pmi,
    "iva": _fetch_iva,
    "cpi": _fetch_cpi,
    "ppi": _fetch_ppi,
    "m2": _fetch_m2,
    "shrzgm": _fetch_shrzgm,
    "shibor": _fetch_shibor,
    "cn10y": _fetch_cn10y,
    "us10y": _fetch_us10y,
    "credit_spread": _fetch_credit_spread,
    "dxy": _fetch_dxy,
}


def get_macro_raw(source_key):
    """返回单个宏观指标原始值 Series（index=发布所属期日期，未做滞后处理）。"""
    fetcher = _MACRO_FETCHERS[source_key]
    df = _cached(f"macro_{source_key}", fetcher)
    return df["value"].rename(source_key)
