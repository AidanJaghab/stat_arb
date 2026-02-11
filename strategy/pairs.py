"""
Pairs-trading strategy: cointegration scan + z-score signal generation.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# --------------------------------------------------------------------------- #
#  Pair discovery
# --------------------------------------------------------------------------- #

def find_cointegrated_pairs(
    prices: pd.DataFrame,
    p_threshold: float = 0.05,
) -> list[dict]:
    """
    Test every unique pair for cointegration (Engle-Granger).

    Returns a list of dicts:
        {"ticker_a": str, "ticker_b": str, "pvalue": float, "hedge_ratio": float}
    sorted by ascending p-value.
    """
    tickers = prices.columns.tolist()
    pairs = []

    for a, b in combinations(tickers, 2):
        series_a = prices[a].values
        series_b = prices[b].values

        # Engle-Granger test
        _, pvalue, _ = coint(series_a, series_b)

        if pvalue < p_threshold:
            # OLS hedge ratio: a = alpha + beta * b + eps
            model = OLS(series_a, add_constant(series_b)).fit()
            hedge_ratio = model.params[1]
            pairs.append({
                "ticker_a": a,
                "ticker_b": b,
                "pvalue": pvalue,
                "hedge_ratio": hedge_ratio,
            })

    pairs.sort(key=lambda p: p["pvalue"])
    return pairs


# --------------------------------------------------------------------------- #
#  Signal generation
# --------------------------------------------------------------------------- #

def compute_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
) -> pd.Series:
    """Spread = price_a - hedge_ratio * price_b."""
    return prices[ticker_a] - hedge_ratio * prices[ticker_b]


def compute_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """Rolling z-score of the spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std


def generate_signals(
    prices: pd.DataFrame,
    pairs: list[dict],
    zscore_lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """
    For each cointegrated pair, produce a position signal per day.

    Signal convention (from the perspective of the spread = A - h*B):
        +1  → long spread  (long A, short B)
        -1  → short spread (short A, long B)
         0  → flat

    Returns a DataFrame with one column per pair label ("A/B") and
    DatetimeIndex matching *prices*.
    """
    signals = pd.DataFrame(index=prices.index)

    for pair in pairs:
        a, b = pair["ticker_a"], pair["ticker_b"]
        hr = pair["hedge_ratio"]
        label = f"{a}/{b}"

        spread = compute_spread(prices, a, b, hr)
        z = compute_zscore(spread, zscore_lookback)

        # State machine: track position for each pair
        pos = pd.Series(0.0, index=prices.index)
        current = 0.0
        for i in range(len(z)):
            if np.isnan(z.iloc[i]):
                pos.iloc[i] = 0.0
                continue

            if current == 0:
                if z.iloc[i] <= -entry_z:
                    current = 1.0   # long spread
                elif z.iloc[i] >= entry_z:
                    current = -1.0  # short spread
            else:
                if abs(z.iloc[i]) <= exit_z:
                    current = 0.0   # exit

            pos.iloc[i] = current

        signals[label] = pos

    return signals
