#!/usr/bin/env python3
"""
Live price feed — fetches current prices for ~1000 stocks every 5 minutes
using yfinance (free, no API key needed).

Saves snapshots to live_feed/prices_latest.csv (overwritten each tick)
and appends to live_feed/prices_history.csv (cumulative log).
Auto-commits and pushes to GitHub after each tick.

Usage:
    python -m live_feed.fetcher
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.universe import get_top_universe

INTERVAL_SECONDS = 300  # 5 minutes
OUTPUT_DIR = Path(__file__).resolve().parent
LATEST_FILE = OUTPUT_DIR / "prices_latest.csv"
HISTORY_FILE = OUTPUT_DIR / "prices_history.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 100


def fetch_current_prices(tickers: list[str]) -> pd.DataFrame:
    """Fetch latest prices via yfinance in batches."""
    all_prices = {}

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        try:
            df = yf.download(
                batch, period="1d", interval="1m",
                auto_adjust=True, progress=False,
            )
            if df.empty:
                continue

            closes = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
            latest = closes.iloc[-1]
            all_prices.update(latest.dropna().to_dict())
        except Exception as e:
            print(f"\n  Warning: batch {i}-{i+len(batch)} failed: {e}", flush=True)

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
    """Main loop: fetch prices every 5 minutes."""
    print("=" * 50)
    print("  LIVE PRICE FEED (yfinance)")
    print("=" * 50)
    print(f"  Interval: {INTERVAL_SECONDS}s ({INTERVAL_SECONDS // 60} min)")
    print(f"  Output:   {LATEST_FILE}")
    print(f"  History:  {HISTORY_FILE}")
    print()

    print("Fetching ticker universe...")
    tickers = get_top_universe(target=1000)
    print(f"Tracking {len(tickers)} tickers.\n")

    tick = 0
    while True:
        tick += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Tick #{tick} — fetching prices...", end=" ", flush=True)

        row = fetch_current_prices(tickers)

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
