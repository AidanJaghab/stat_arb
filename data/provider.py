"""
Data provider interface and implementations.

To add a new provider:
  1. Subclass DataProvider
  2. Implement get_prices()
  3. Register it in get_provider()
  4. Set PROVIDER in config.py
"""

import os
from abc import ABC, abstractmethod

import pandas as pd
import yfinance as yf
import databento as db


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


class DatabentoProvider(DataProvider):
    """Fetch data via the Databento API. Requires DATABENTO_API_KEY env var."""

    def __init__(self) -> None:
        key = os.environ.get("DATABENTO_API_KEY")
        if not key:
            raise RuntimeError(
                "Set DATABENTO_API_KEY environment variable to use DatabentoProvider"
            )
        self.client = db.Historical(key)

    def get_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        # Databento uses dataset + schema; XNAS.ITCH for US equities
        data = self.client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=tickers,
            schema="ohlcv-1d",
            start=start,
            end=end,
        )
        df = data.to_df()
        # Pivot to (dates x tickers) using the close column
        df = df.reset_index()
        pivot = df.pivot_table(
            index="ts_event", columns="symbol", values="close"
        )
        pivot.index = pd.to_datetime(pivot.index).tz_localize(None)
        pivot.index.name = "Date"
        return pivot.dropna(axis=1, how="all").dropna()


# --- Factory --------------------------------------------------------------- #

def get_provider(name: str = "yfinance") -> DataProvider:
    """Return the provider instance matching *name*."""
    providers = {
        "yfinance": YFinanceProvider,
        "databento": DatabentoProvider,
        # "polygon": PolygonProvider,
        # "alpaca":  AlpacaProvider,
    }
    if name not in providers:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(providers)}"
        )
    return providers[name]()
