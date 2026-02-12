#!/usr/bin/env python3
"""
Live price feed — fetches current prices for ~1000 stocks every 5 minutes
using the Databento API.

Saves snapshots to live_feed/prices_latest.csv (overwritten each tick)
and appends to live_feed/prices_history.csv (cumulative log).
Auto-commits and pushes to GitHub after each tick.

Usage:
    export DATABENTO_API_KEY="db-..."
    python -m live_feed.fetcher
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import databento as db
import pandas as pd

# Add project root to path so we can import data.universe
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.universe import get_top_universe

INTERVAL_SECONDS = 300  # 5 minutes
OUTPUT_DIR = Path(__file__).resolve().parent
LATEST_FILE = OUTPUT_DIR / "prices_latest.csv"
HISTORY_FILE = OUTPUT_DIR / "prices_history.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_databento_client() -> db.Historical:
    """Initialize Databento client from env var."""
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("ERROR: Set DATABENTO_API_KEY environment variable.")
        sys.exit(1)
    return db.Historical(key)


def fetch_current_prices(client: db.Historical, tickers: list[str]) -> pd.DataFrame:
    """
    Fetch the latest prices for all tickers via Databento.
    Returns a single-row DataFrame: columns = tickers, index = timestamp.
    """
    now = datetime.utcnow()
    start = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M")
    end = now.strftime("%Y-%m-%dT%H:%M")

    all_prices = {}

    # Databento supports batched symbol queries
    batch_size = 200
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=batch,
                schema="ohlcv-1m",
                start=start,
                end=end,
            )
            df = data.to_df()
            if df.empty:
                continue

            df = df.reset_index()
            # Get the last close per symbol
            latest = df.groupby("symbol")["close"].last()
            all_prices.update(latest.to_dict())

        except Exception as e:
            # Try XNYS (NYSE) for tickers not on NASDAQ
            try:
                data = client.timeseries.get_range(
                    dataset="XNYS.TRADES",
                    symbols=batch,
                    schema="ohlcv-1m",
                    start=start,
                    end=end,
                )
                df = data.to_df()
                if not df.empty:
                    df = df.reset_index()
                    latest = df.groupby("symbol")["close"].last()
                    all_prices.update(latest.to_dict())
            except Exception as e2:
                print(f"\n  Warning: batch {i}-{i+len(batch)} failed: {e2}", flush=True)
                continue

    if not all_prices:
        return pd.DataFrame()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([all_prices])
    row.index = [timestamp]
    row.index.name = "timestamp"
    return row


def save_snapshot(row: pd.DataFrame) -> None:
    """Save latest prices and append to history."""
    row.to_csv(LATEST_FILE)

    if HISTORY_FILE.exists():
        row.to_csv(HISTORY_FILE, mode="a", header=False)
    else:
        row.to_csv(HISTORY_FILE)


def git_push(tick: int) -> None:
    """Commit and push price data to GitHub."""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cmds = [
            ["git", "add", "live_feed/prices_latest.csv", "live_feed/prices_history.csv"],
            ["git", "commit", "-m", f"price update #{tick} — {now}"],
            ["git", "push", "origin", "main"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, timeout=30)
        print("  pushed to GitHub.", flush=True)
    except Exception as e:
        print(f"  git push failed: {e}", flush=True)


def run_live_feed() -> None:
    """Main loop: fetch prices every 5 minutes via Databento."""
    print("=" * 50)
    print("  LIVE PRICE FEED (Databento)")
    print("=" * 50)
    print(f"  Interval: {INTERVAL_SECONDS}s ({INTERVAL_SECONDS // 60} min)")
    print(f"  Output:   {LATEST_FILE}")
    print(f"  History:  {HISTORY_FILE}")
    print()

    client = get_databento_client()
    print("Databento client initialized.")

    print("Fetching ticker universe...")
    tickers = get_top_universe(target=1000)
    print(f"Tracking {len(tickers)} tickers.\n")

    tick = 0
    while True:
        tick += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Tick #{tick} — fetching prices...", end=" ", flush=True)

        row = fetch_current_prices(client, tickers)

        if row.empty:
            print("no data (market may be closed)")
        else:
            save_snapshot(row)
            n_prices = row.notna().sum(axis=1).iloc[0]
            print(f"got {int(n_prices)}/{len(tickers)} prices, saved.", flush=True)
            git_push(tick)

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_live_feed()
