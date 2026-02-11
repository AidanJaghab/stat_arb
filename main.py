#!/usr/bin/env python3
"""
Equity Statistical Arbitrage — end-to-end runner.

Usage:
    python main.py
"""

import sys
import config
from data.provider import get_provider
from backtest.engine import run_backtest
from metrics.performance import compute_metrics, print_metrics, plot_equity_curve


def main() -> None:
    print("=" * 50)
    print("  EQUITY STAT-ARB SYSTEM")
    print("=" * 50)

    # --- 1. Load data ---
    print(f"\n[1/4] Fetching price data via '{config.PROVIDER}' ...")
    provider = get_provider(config.PROVIDER)

    all_tickers = config.TICKERS + [config.BENCHMARK]
    prices = provider.get_prices(all_tickers, config.START_DATE, config.END_DATE)

    # Separate benchmark
    if config.BENCHMARK in prices.columns:
        benchmark_prices = prices[config.BENCHMARK]
        asset_prices = prices.drop(columns=[config.BENCHMARK])
    else:
        benchmark_prices = None
        asset_prices = prices

    print(f"    {len(asset_prices.columns)} tickers, "
          f"{len(asset_prices)} trading days loaded.")

    # --- 2. Run walk-forward backtest ---
    print("\n[2/4] Running walk-forward backtest ...")
    results = run_backtest(
        asset_prices,
        training_window=config.TRAINING_WINDOW,
        trading_window=config.TRADING_WINDOW,
        coint_pvalue=config.COINT_PVALUE,
        zscore_lookback=config.ZSCORE_LOOKBACK,
        entry_z=config.ZSCORE_ENTRY,
        exit_z=config.ZSCORE_EXIT,
        max_position_weight=config.MAX_POSITION_WEIGHT,
        max_gross_leverage=config.MAX_GROSS_LEVERAGE,
    )

    if results.empty:
        print("No trades generated. Try widening the universe or relaxing thresholds.")
        sys.exit(1)

    # --- 3. Compute & display metrics ---
    print("\n[3/4] Computing performance metrics ...")
    metrics = compute_metrics(results, benchmark_prices)
    print_metrics(metrics)

    # --- 4. Plot equity curve ---
    print("[4/4] Plotting equity curve ...")
    plot_equity_curve(results, save_path="equity_curve.png")

    print("Done.\n")


if __name__ == "__main__":
    main()
