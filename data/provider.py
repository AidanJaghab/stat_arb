"""
Data provider interface and implementations.

To add a new provider (e.g. Polygon, Alpaca):
  1. Subclass DataProvider
  2. Implement get_prices()
  3. Set PROVIDER in config.py
"""

from abc import ABC, abstractmethod

import pandas as pd
import yfinance as yf


class DataProvider(ABC):
    """Abstract interface for fetching adjusted close prices."""

    @abstractmethod
    def get_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return a DataFrame of adjusted close prices (dates x tickers)."""
        ...


class YFinanceProvider(DataProvider):
    """Fetch data via the yfinance library (free, no API key)."""

    def get_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        df = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
        # yf.download returns MultiIndex columns when len(tickers) > 1
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        return df.dropna(axis=1, how="all").dropna()


# --- Factory --------------------------------------------------------------- #

def get_provider(name: str = "yfinance") -> DataProvider:
    """Return the provider instance matching *name*."""
    providers = {
        "yfinance": YFinanceProvider,
        # "polygon": PolygonProvider,
        # "alpaca":  AlpacaProvider,
    }
    if name not in providers:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(providers)}"
        )
    return providers[name]()
