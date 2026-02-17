"""
Configuration for the stat-arb system.
Edit these values or override them at runtime.
"""

from data.universe import get_top_universe

# --- Universe ---
# Dynamically fetch top ~1000 US equities (S&P 500 + S&P 400 + extras)
TICKERS = get_top_universe(target=1000)

BENCHMARK = "SPY"

# --- Date range ---
START_DATE = "2024-01-01"
END_DATE = "2026-02-12"

# --- Cointegration / pairs ---
COINT_PVALUE = 0.05          # max p-value to accept a pair
ZSCORE_LOOKBACK = 20         # rolling z-score window (trading days)
ZSCORE_ENTRY = 2.0           # enter when |z| >= this
ZSCORE_EXIT = 0.5            # exit when |z| <= this

# --- Walk-forward windows (trading days) ---
TRAINING_WINDOW = 252        # ~1 year
TRADING_WINDOW = 21          # ~1 month

# --- Risk limits ---
MAX_POSITION_WEIGHT = 0.05   # 5% of capital per leg
MAX_GROSS_LEVERAGE = 4.0     # sum of |weights|

# --- Data provider ---
PROVIDER = "alpaca"          # uses ALPACA_API_KEY env var
