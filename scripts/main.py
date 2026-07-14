"""一键运行：数据抓取 -> 宏观因子 -> 冲击提取 -> 三策略 x 5档cap回测 -> 绩效表与图表输出。"""

import os
import pandas as pd

import config
import backtest


def main():
    print("=== 开始回测：宏观冲击风险平价模型（Macro RP + Joint RP）===")
    results, prices = backtest.run_backtest()
    print(f"共完成 {len(results)} 个 (策略, cap) 组合的回测。")

    perf_df = backtest.build_performance_dataframe(results)
    perf_path = os.path.join(config.OUTPUT_DIR, "performance_table.csv")
    perf_df.to_csv(perf_path, index=False, encoding="utf-8-sig")
    print(f"绩效表已保存: {perf_path}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(perf_df[["策略", "cap", "年化收益", "年化波动", "Sharpe", "最大回撤", "Calmar", "日均换手"]])

    for cap in config.CAP_LEVELS:
        nav_path = os.path.join(config.OUTPUT_DIR, f"nav_comparison_cap{cap}.png")
        backtest.plot_nav_comparison(results, cap, nav_path)

    excess_path = os.path.join(config.OUTPUT_DIR, f"yearly_excess_cap{config.DEFAULT_CAP_FOR_PLOT}.png")
    backtest.plot_yearly_excess(results, config.DEFAULT_CAP_FOR_PLOT, excess_path)

    holdings_path = os.path.join(config.OUTPUT_DIR, f"joint_rp_holdings_cap{config.DEFAULT_CAP_FOR_PLOT}.png")
    backtest.plot_joint_holdings(results, config.DEFAULT_CAP_FOR_PLOT, holdings_path)

    print(f"图表已保存至: {config.OUTPUT_DIR}")
    print("=== 回测完成 ===")


if __name__ == "__main__":
    main()
