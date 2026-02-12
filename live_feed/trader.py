#!/usr/bin/env python3
"""
Live 5-minute trading signal engine.

Every 5 minutes:
  1. Fetch latest 5-min bars for the active pairs
  2. Recompute z-scores
  3. Generate entry/exit signals
  4. Output trade recommendations with position sizing
  5. Log signals and push to GitHub

Usage:
    python -m live_feed.trader
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INTERVAL_SECONDS = 300  # 5 minutes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_FILE = PROJECT_ROOT / "live_feed" / "signals.csv"
POSITIONS_FILE = PROJECT_ROOT / "live_feed" / "positions.csv"

# --- Strategy parameters ---
ZSCORE_LOOKBACK = 60       # 60 x 5min = 5 hours rolling window
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
TOTAL_CAPITAL = 1_000_000  # notional capital for sizing
MAX_PAIRS = 10
MAX_ALLOC_PER_PAIR = 0.10  # 10% gross per pair


class PairPosition:
    """Track state for a single pair."""

    def __init__(self, ticker_a: str, ticker_b: str, hedge_ratio: float, sector: str):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = hedge_ratio
        self.sector = sector
        self.signal = 0  # +1 long spread, -1 short spread, 0 flat
        self.entry_z = None
        self.entry_time = None

    def update(self, z_score: float, timestamp: str) -> dict | None:
        """
        Update position based on z-score.
        Returns a trade action dict if a signal fires, else None.
        """
        prev_signal = self.signal
        action = None

        if self.signal == 0:
            # Look for entry
            if z_score <= -ZSCORE_ENTRY:
                self.signal = 1  # long spread: long A, short B
                self.entry_z = z_score
                self.entry_time = timestamp
                action = {
                    "action": "ENTER_LONG_SPREAD",
                    "long": self.ticker_a,
                    "short": self.ticker_b,
                }
            elif z_score >= ZSCORE_ENTRY:
                self.signal = -1  # short spread: short A, long B
                self.entry_z = z_score
                self.entry_time = timestamp
                action = {
                    "action": "ENTER_SHORT_SPREAD",
                    "long": self.ticker_b,
                    "short": self.ticker_a,
                }
        else:
            # Look for exit
            if abs(z_score) <= ZSCORE_EXIT:
                action = {
                    "action": "EXIT",
                    "prev_signal": "LONG_SPREAD" if self.signal == 1 else "SHORT_SPREAD",
                    "entry_z": self.entry_z,
                    "exit_z": z_score,
                    "bars_held": None,  # filled by caller
                }
                self.signal = 0
                self.entry_z = None
                self.entry_time = None

        if action:
            action.update({
                "pair": f"{self.ticker_a}/{self.ticker_b}",
                "hedge_ratio": self.hedge_ratio,
                "z_score": z_score,
                "timestamp": timestamp,
                "sector": self.sector,
            })

        return action


def load_pairs() -> list[dict]:
    """
    Load pairs from top_pairs.txt or run scanner if not found.
    Returns list of pair dicts with ticker_a, ticker_b, hedge_ratio, sector.
    """
    pairs_file = PROJECT_ROOT / "top_pairs.txt"
    pairs_csv = PROJECT_ROOT / "live_feed" / "active_pairs.csv"

    if pairs_csv.exists():
        df = pd.read_csv(pairs_csv)
        return df.to_dict("records")

    # If no pre-computed pairs, use these well-known stat-arb pairs as defaults
    # (scanner should be run first for proper pair selection)
    print("WARNING: No active_pairs.csv found. Run 'python -m strategy.scanner' first.")
    print("Using fallback pairs...\n")
    return [
        {"ticker_a": "KO", "ticker_b": "PEP", "hedge_ratio": 1.0, "sector": "Consumer Staples"},
        {"ticker_a": "XOM", "ticker_b": "CVX", "hedge_ratio": 1.0, "sector": "Energy"},
        {"ticker_a": "GS", "ticker_b": "MS", "hedge_ratio": 1.0, "sector": "Financials"},
        {"ticker_a": "JPM", "ticker_b": "BAC", "hedge_ratio": 1.0, "sector": "Financials"},
        {"ticker_a": "HD", "ticker_b": "LOW", "hedge_ratio": 1.0, "sector": "Consumer Discretionary"},
    ]


def fetch_5min_data(tickers: list[str]) -> pd.DataFrame:
    """Fetch recent 5-min bars for the given tickers."""
    df = yf.download(
        tickers, period="5d", interval="5m",
        auto_adjust=True, progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna()


def compute_zscore(spread: pd.Series, lookback: int = ZSCORE_LOOKBACK) -> float:
    """Compute current z-score of the spread."""
    if len(spread) < lookback:
        lookback = len(spread)
    recent = spread.iloc[-lookback:]
    mean = recent.mean()
    std = recent.std()
    if std < 1e-8:
        return 0.0
    return (spread.iloc[-1] - mean) / std


def format_signal_table(positions: list[PairPosition], z_scores: dict) -> str:
    """Format current pair status as a table."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n{'='*70}")
    lines.append(f"  LIVE SIGNALS — {now}")
    lines.append(f"{'='*70}")
    lines.append(f"  {'Pair':<15} {'Z-Score':>8} {'Signal':>15} {'Alloc':>8}")
    lines.append(f"  {'-'*15} {'-'*8} {'-'*15} {'-'*8}")

    for pos in positions:
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        z = z_scores.get(label, 0)
        if pos.signal == 1:
            sig = f"LONG {pos.ticker_a}"
        elif pos.signal == -1:
            sig = f"LONG {pos.ticker_b}"
        else:
            sig = "FLAT"

        alloc = f"{MAX_ALLOC_PER_PAIR*100:.0f}%" if pos.signal != 0 else "—"
        lines.append(f"  {label:<15} {z:>+8.2f} {sig:>15} {alloc:>8}")

    active = sum(1 for p in positions if p.signal != 0)
    gross = active * MAX_ALLOC_PER_PAIR * 2 * 100
    lines.append(f"\n  Active pairs: {active}/{len(positions)}")
    lines.append(f"  Gross exposure: {gross:.0f}%")
    lines.append(f"  Net exposure: ~0% (market neutral)")
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def log_signal(action: dict) -> None:
    """Append trade signal to signals CSV."""
    row = pd.DataFrame([action])
    if SIGNALS_FILE.exists():
        row.to_csv(SIGNALS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(SIGNALS_FILE, index=False)


def git_push(msg: str) -> None:
    """Commit and push signal/position data."""
    try:
        files = [
            "live_feed/signals.csv",
            "live_feed/positions.csv",
        ]
        cmds = [
            ["git", "add"] + files,
            ["git", "commit", "-m", msg],
            ["git", "push", "origin", "main"],
        ]
        for cmd in cmds:
            subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, timeout=30)
    except Exception:
        pass


def run_trader() -> None:
    """Main 5-minute trading loop."""
    print("=" * 60)
    print("  LIVE STAT-ARB TRADER (5-min)")
    print("=" * 60)

    pair_configs = load_pairs()
    print(f"Loaded {len(pair_configs)} pairs.\n")

    # Initialize position trackers
    positions = []
    all_tickers = set()
    for p in pair_configs:
        pos = PairPosition(p["ticker_a"], p["ticker_b"], p["hedge_ratio"], p["sector"])
        positions.append(pos)
        all_tickers.update([p["ticker_a"], p["ticker_b"]])

    all_tickers = sorted(all_tickers)
    print(f"Tracking {len(all_tickers)} unique tickers across {len(positions)} pairs.\n")

    tick = 0
    while True:
        tick += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Tick #{tick}", flush=True)

        # Fetch latest 5-min data
        try:
            prices = fetch_5min_data(all_tickers)
        except Exception as e:
            print(f"  Data fetch failed: {e}", flush=True)
            time.sleep(INTERVAL_SECONDS)
            continue

        if prices.empty:
            print("  No data (market may be closed)", flush=True)
            time.sleep(INTERVAL_SECONDS)
            continue

        # Compute z-scores and update signals
        z_scores = {}
        actions = []
        for pos in positions:
            if pos.ticker_a not in prices.columns or pos.ticker_b not in prices.columns:
                continue

            spread = prices[pos.ticker_a] - pos.hedge_ratio * prices[pos.ticker_b]
            z = compute_zscore(spread)
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            z_scores[label] = z

            action = pos.update(z, now)
            if action:
                actions.append(action)
                log_signal(action)
                print(f"  >>> {action['action']}: {action['pair']} @ z={z:+.2f}", flush=True)

        # Print status table
        table = format_signal_table(positions, z_scores)
        print(table, flush=True)

        # Save current positions
        pos_data = []
        for pos in positions:
            label = f"{pos.ticker_a}/{pos.ticker_b}"
            pos_data.append({
                "pair": label,
                "signal": pos.signal,
                "z_score": z_scores.get(label, 0),
                "hedge_ratio": pos.hedge_ratio,
                "sector": pos.sector,
                "entry_time": pos.entry_time,
            })
        pd.DataFrame(pos_data).to_csv(POSITIONS_FILE, index=False)

        # Push to GitHub
        if actions:
            git_push(f"trade signal tick #{tick} — {now}")
        else:
            git_push(f"position update #{tick} — {now}")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_trader()
