#!/usr/bin/env python3
"""
Alpaca client for paper trading and market data.

Provides helpers to:
  - Fetch 5-min bar data (replacing yfinance)
  - Execute paper trades (market orders)
  - Query account info and positions
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

_trading_client = None
_data_client = None


def _get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        _trading_client = TradingClient(api_key, secret_key, paper=True)
    return _trading_client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        _data_client = StockHistoricalDataClient(api_key, secret_key)
    return _data_client


def fetch_5min_data_alpaca(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch recent 1-min bars for given tickers via Alpaca.

    Returns a DataFrame with DatetimeIndex and one column per ticker (close prices),
    matching the format previously returned by yfinance.
    """
    end = datetime.now()
    start = end - timedelta(days=2)  # ~156 bars per ticker at 5-min, enough for lookback

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
    )

    bars = _get_data_client().get_stock_bars(request)
    bar_df = bars.df  # MultiIndex: (symbol, timestamp)

    if bar_df.empty:
        return pd.DataFrame()

    # Pivot to get one column per ticker with close prices
    bar_df = bar_df.reset_index()
    pivot = bar_df.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot.ffill().dropna(how="all")


def fetch_latest_prices_alpaca(tickers: list[str], batch_size: int = 100) -> dict:
    """
    Fetch the latest price for each ticker via Alpaca 1-min bars.

    Returns dict of {ticker: price}.
    """
    all_prices = {}
    end = datetime.now()
    start = end - timedelta(minutes=30)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = _get_data_client().get_stock_bars(request)
            bar_df = bars.df

            if bar_df.empty:
                continue

            bar_df = bar_df.reset_index()
            for ticker in batch:
                ticker_data = bar_df[bar_df["symbol"] == ticker]
                if not ticker_data.empty:
                    all_prices[ticker] = ticker_data["close"].iloc[-1]
        except Exception as e:
            print(f"  Warning: batch {i}-{i+len(batch)} failed: {e}", flush=True)

    return all_prices


def execute_trade(action: dict) -> None:
    """
    Execute a paper trade via Alpaca based on the action dict from PairPosition.

    Supported actions:
      - ENTER_LONG_SPREAD / ENTER_SHORT_SPREAD: buy 'long' ticker, sell short 'short' ticker
      - EXIT: close both legs of the pair
    """
    act = action.get("action", "")
    pair = action.get("pair", "")

    if act in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD"):
        long_ticker = action["long"]
        short_ticker = action["short"]
        shares_long = action.get("shares_long", action.get("shares_a", 0))
        shares_short = action.get("shares_short", action.get("shares_b", 0))

        if shares_long > 0:
            _submit_order(long_ticker, shares_long, OrderSide.BUY)
        if shares_short > 0:
            _submit_order(short_ticker, shares_short, OrderSide.SELL)

        print(f"  [ALPACA] Entered {act}: BUY {shares_long} {long_ticker}, "
              f"SELL {shares_short} {short_ticker}", flush=True)

    elif act == "EXIT":
        # Close both legs — figure out tickers from pair string "A/B"
        tickers = pair.split("/")
        if len(tickers) == 2:
            _close_position(tickers[0])
            _close_position(tickers[1])
            print(f"  [ALPACA] Exited pair {pair}: closed positions in "
                  f"{tickers[0]} and {tickers[1]}", flush=True)


def _submit_order(ticker: str, qty: int, side: OrderSide) -> None:
    """Submit a market order."""
    try:
        order = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        _get_trading_client().submit_order(order)
    except Exception as e:
        print(f"  [ALPACA] Order failed ({side.name} {qty} {ticker}): {e}", flush=True)


def _close_position(ticker: str) -> None:
    """Close any existing position in the given ticker."""
    try:
        _get_trading_client().close_position(ticker)
    except Exception as e:
        # No position to close is fine
        if "position does not exist" not in str(e).lower():
            print(f"  [ALPACA] Close position failed ({ticker}): {e}", flush=True)


def get_account_info() -> dict:
    """Fetch paper account balance and key info."""
    account = _get_trading_client().get_account()
    return {
        "equity": account.equity,
        "cash": account.cash,
        "buying_power": account.buying_power,
        "portfolio_value": account.portfolio_value,
        "status": account.status,
    }


def get_all_tradeable_tickers() -> list[str]:
    """Get all active, tradeable US equity tickers from Alpaca (NASDAQ + NYSE)."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = _get_trading_client().get_all_assets(request)
    tickers = [
        a.symbol for a in assets
        if a.tradable and a.exchange in ("NASDAQ", "NYSE")
        and a.shortable  # need shortable for stat arb
        and not a.symbol.isdigit()  # skip weird tickers
        and "." not in a.symbol  # skip preferred shares etc.
        and len(a.symbol) <= 5  # skip long tickers (warrants, units)
    ]
    return sorted(tickers)


def fetch_5min_data_alpaca_batch(
    tickers: list[str], days: int = 5, batch_size: int = 500,
) -> pd.DataFrame:
    """
    Fetch 5-min bars for a large set of tickers in batches.
    Returns DataFrame with DatetimeIndex x ticker columns (close prices).
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    all_frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
                end=end,
            )
            bars = _get_data_client().get_stock_bars(request)
            bar_df = bars.df
            if not bar_df.empty:
                all_frames.append(bar_df)
            print(f"  Fetched batch {i}-{i+len(batch)} "
                  f"({len(bar_df) if not bar_df.empty else 0} bars)", flush=True)
        except Exception as e:
            print(f"  Warning: batch {i}-{i+len(batch)} failed: {e}", flush=True)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined.reset_index()
    pivot = combined.pivot_table(index="timestamp", columns="symbol", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot


def get_positions() -> list[dict]:
    """Fetch all current Alpaca positions."""
    positions = _get_trading_client().get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": p.qty,
            "side": p.side,
            "market_value": p.market_value,
            "unrealized_pl": p.unrealized_pl,
        }
        for p in positions
    ]
