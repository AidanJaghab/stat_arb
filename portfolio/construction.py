"""
Dollar-neutral portfolio construction with risk limits.
"""

import numpy as np
import pandas as pd


def build_weights(
    signals: pd.DataFrame,
    pairs: list[dict],
    max_position_weight: float = 0.05,
    max_gross_leverage: float = 4.0,
) -> pd.DataFrame:
    """
    Convert pair signals into per-ticker dollar weights.

    For a pair A/B with hedge ratio h and signal s:
        weight_A = +s * w
        weight_B = -s * h * w
    where w is capped at max_position_weight.

    After stacking all pairs, weights are rescaled so that
    gross leverage = sum(|weights|) does not exceed max_gross_leverage.

    Returns a DataFrame (dates x tickers) of portfolio weights.
    """
    # Map pair label -> pair dict for quick lookup
    pair_map = {f"{p['ticker_a']}/{p['ticker_b']}": p for p in pairs}

    # Collect all tickers that appear in any active pair
    all_tickers = set()
    for p in pairs:
        all_tickers.update([p["ticker_a"], p["ticker_b"]])
    all_tickers = sorted(all_tickers)

    weights = pd.DataFrame(0.0, index=signals.index, columns=all_tickers)

    for label in signals.columns:
        p = pair_map[label]
        a, b = p["ticker_a"], p["ticker_b"]
        hr = p["hedge_ratio"]
        sig = signals[label]

        # Raw weight per leg (capped at max_position_weight)
        w = np.minimum(max_position_weight, max_position_weight)  # base weight
        weights[a] += sig * w
        weights[b] += -sig * hr * w

    # --- Enforce risk limits ------------------------------------------------ #

    # 1. Cap individual positions
    weights = weights.clip(-max_position_weight, max_position_weight)

    # 2. Rescale if gross leverage exceeds limit
    gross = weights.abs().sum(axis=1)
    scale = (max_gross_leverage / gross).clip(upper=1.0)
    weights = weights.mul(scale, axis=0)

    return weights
