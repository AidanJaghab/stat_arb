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
from datetime import datetime

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class DataProvider(ABC):
    """Abstract interface for fetching adjusted close prices."""

    @abstractmethod
    def get_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Return a DataFrame of adjusted close prices (dates x tickers)."""
        ...


class AlpacaProvider(DataProvider):
    """Fetch daily bar data via Alpaca. Uses ALPACA_API_KEY / ALPACA_SECRET_KEY env vars."""

    def __init__(self) -> None:
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def get_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        batch_size = 500
        all_frames = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=datetime.strptime(start, "%Y-%m-%d"),
                    end=datetime.strptime(end, "%Y-%m-%d"),
                )
                bars = self.client.get_stock_bars(request)
                bar_df = bars.df
                if not bar_df.empty:
                    all_frames.append(bar_df)
                print(f"    Fetched batch {i}-{i+len(batch)} "
                      f"({len(bar_df)} bars)", flush=True)
            except Exception as e:
                print(f"    Warning: batch {i}-{i+len(batch)} failed: {e}",
                      flush=True)

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames).reset_index()
        pivot = combined.pivot_table(
            index="timestamp", columns="symbol", values="close"
        )
        pivot.index = pd.to_datetime(pivot.index).tz_localize(None)
        pivot.index.name = "Date"
        # Drop tickers with insufficient data, then forward-fill small gaps
        min_rows = len(pivot) * 0.5
        valid = [c for c in pivot.columns if pivot[c].notna().sum() >= min_rows]
        pivot = pivot[valid].ffill().dropna()
        return pivot


# --- Factory --------------------------------------------------------------- #

def get_provider(name: str = "alpaca") -> DataProvider:
    """Return the provider instance matching *name*."""
    providers = {
        "alpaca": AlpacaProvider,
    }
    if name not in providers:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(providers)}"
        )
    return providers[name]()
