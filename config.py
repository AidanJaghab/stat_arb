"""
Configuration for the stat-arb system.
Edit these values or override them at runtime.
"""

# --- Universe ---
# Top liquid S&P 500 names (keep small for fast MVP; expand as needed)
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "UNH", "PG", "HD", "MA", "DIS", "BAC", "XOM",
    "PFE", "CSCO", "ADBE", "CRM", "ABT", "CVX", "KO", "PEP", "TMO",
    "COST", "AVGO", "MRK", "WMT", "LLY", "ACN", "MCD", "NEE", "LIN",
    "TXN", "DHR", "PM", "BMY", "AMGN", "HON", "UPS", "CAT", "GS",
    "LOW", "SBUX", "BLK", "ISRG", "MDT",
]

BENCHMARK = "SPY"

# --- Date range ---
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"

# --- Cointegration / pairs ---
COINT_PVALUE = 0.05          # max p-value to accept a pair
ZSCORE_LOOKBACK = 20         # rolling z-score window (trading days)
ZSCORE_ENTRY = 2.0           # enter when |z| >= this
ZSCORE_EXIT = 0.5            # exit when |z| <= this

# --- Walk-forward windows (trading days) ---
TRAINING_WINDOW = 252        # ~1 year
TRADING_WINDOW = 63          # ~1 quarter

# --- Risk limits ---
MAX_POSITION_WEIGHT = 0.05   # 5% of capital per leg
MAX_GROSS_LEVERAGE = 4.0     # sum of |weights|

# --- Data provider ---
PROVIDER = "yfinance"        # swap to "polygon" or "alpaca" when ready
