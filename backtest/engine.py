"""
Walk-forward backtest engine.

Flow:
  1. Split the price history into rolling (train, trade) windows.
  2. On each window:
     a. Find cointegrated pairs on the training data.
     b. Generate signals on the trading data.
     c. Build portfolio weights.
  3. Stitch together daily portfolio returns across all windows.
"""

import pandas as pd

from strategy.pairs import find_cointegrated_pairs, generate_signals
from portfolio.construction import build_weights


def run_backtest(
    prices: pd.DataFrame,
    training_window: int = 252,
    trading_window: int = 63,
    coint_pvalue: float = 0.05,
    zscore_lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    max_position_weight: float = 0.05,
    max_gross_leverage: float = 4.0,
) -> pd.DataFrame:
    """
    Run a walk-forward backtest.

    Returns a DataFrame with columns:
        portfolio_return  – daily dollar-weighted return
        gross_leverage    – sum of |weights|
        net_exposure      – sum of weights (should be ~0)
        n_pairs           – number of active pairs this window
    """
    dates = prices.index
    n = len(dates)
    results = []

    start = 0
    while start + training_window + trading_window <= n:
        train_end = start + training_window
        trade_end = min(train_end + trading_window, n)

        train_prices = prices.iloc[start:train_end]
        trade_prices = prices.iloc[train_end:trade_end]

        # --- Step 1: discover pairs on training data ---
        pairs = find_cointegrated_pairs(train_prices, p_threshold=coint_pvalue)

        if not pairs:
            # No pairs found — record zero returns for this window
            for dt in trade_prices.index:
                results.append({
                    "date": dt,
                    "portfolio_return": 0.0,
                    "gross_leverage": 0.0,
                    "net_exposure": 0.0,
                    "n_pairs": 0,
                })
            start += trading_window
            continue

        # --- Step 2: generate signals on trading data ---
        signals = generate_signals(
            trade_prices, pairs,
            zscore_lookback=zscore_lookback,
            entry_z=entry_z,
            exit_z=exit_z,
        )

        # --- Step 3: portfolio weights ---
        weights = build_weights(
            signals, pairs,
            max_position_weight=max_position_weight,
            max_gross_leverage=max_gross_leverage,
        )

        # --- Step 4: compute daily returns ---
        # Align returns with weights (weights at t drive return from t to t+1)
        asset_returns = trade_prices.pct_change()
        # Only keep tickers present in weights
        common = weights.columns.intersection(asset_returns.columns)
        w = weights[common]
        r = asset_returns[common]

        port_ret = (w.shift(1) * r).sum(axis=1)  # shift weights by 1 day

        for i, dt in enumerate(trade_prices.index):
            results.append({
                "date": dt,
                "portfolio_return": port_ret.iloc[i],
                "gross_leverage": w.iloc[i].abs().sum() if i < len(w) else 0,
                "net_exposure": w.iloc[i].sum() if i < len(w) else 0,
                "n_pairs": len(pairs),
            })

        start += trading_window

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("date")
        # Deduplicate in case of overlapping windows
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
    return df
