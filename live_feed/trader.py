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
OUTPUT_LOG = PROJECT_ROOT / "live_feed" / "trader_output.log"

# --- Strategy parameters ---
ZSCORE_LOOKBACK = 60       # 60 x 5min = 5 hours rolling window
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
TOTAL_CAPITAL = 50_000     # total account size
MAX_PAIRS = 10
MAX_EXPOSURE_PER_PAIR = 2_500  # $2,500 per leg = $5,000 gross per pair, max $25k across 10 pairs


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


def compute_shares(price: float, dollar_amount: float) -> int:
    """Convert dollar amount to whole shares."""
    if price <= 0:
        return 0
    return int(dollar_amount / price)


def format_signal_table(
    positions: list[PairPosition],
    z_scores: dict,
    latest_prices: dict,
) -> str:
    """Format current pair status with exact share counts and dollar amounts."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n{'='*70}")
    lines.append(f"  LIVE SIGNALS — {now}")
    lines.append(f"{'='*70}")

    total_long_dollars = 0
    total_short_dollars = 0

    for pos in positions:
        label = f"{pos.ticker_a}/{pos.ticker_b}"
        z = z_scores.get(label, 0)
        price_a = latest_prices.get(pos.ticker_a, 0)
        price_b = latest_prices.get(pos.ticker_b, 0)

        lines.append(f"\n  PAIR: {pos.ticker_a} / {pos.ticker_b}  ({pos.sector})")
        lines.append(f"  Z-Score: {z:+.2f}")
        lines.append(f"  Prices: {pos.ticker_a} = ${price_a:.2f}  |  {pos.ticker_b} = ${price_b:.2f}")

        if pos.signal == 0:
            # No active position — show what would happen at entry
            if abs(z) < ZSCORE_EXIT:
                lines.append(f"  Status: FLAT — no trade (z within normal range)")
            else:
                lines.append(f"  Status: FLAT — watching (z approaching entry)")

            # Show projected trade at entry
            shares_a = compute_shares(price_a, MAX_EXPOSURE_PER_PAIR)
            shares_b = compute_shares(price_b, MAX_EXPOSURE_PER_PAIR)
            if z > 0:
                lines.append(f"  If z hits +{ZSCORE_ENTRY:.1f} → Short {shares_a} shares {pos.ticker_a} (${shares_a * price_a:,.2f})")
                lines.append(f"                    Buy {shares_b} shares {pos.ticker_b} (${shares_b * price_b:,.2f})")
            else:
                lines.append(f"  If z hits -{ZSCORE_ENTRY:.1f} → Buy {shares_a} shares {pos.ticker_a} (${shares_a * price_a:,.2f})")
                lines.append(f"                    Short {shares_b} shares {pos.ticker_b} (${shares_b * price_b:,.2f})")
        else:
            # Active position — show exact shares
            shares_a = compute_shares(price_a, MAX_EXPOSURE_PER_PAIR)
            shares_b = compute_shares(price_b, MAX_EXPOSURE_PER_PAIR)
            long_dollars = shares_a * price_a if pos.signal == 1 else shares_b * price_b
            short_dollars = shares_b * price_b if pos.signal == 1 else shares_a * price_a

            if pos.signal == 1:
                lines.append(f"  ACTION: LONG SPREAD")
                lines.append(f"    BUY  {shares_a} shares of {pos.ticker_a} @ ${price_a:.2f} = ${shares_a * price_a:,.2f}")
                lines.append(f"    SHORT {shares_b} shares of {pos.ticker_b} @ ${price_b:.2f} = ${shares_b * price_b:,.2f}")
            else:
                lines.append(f"  ACTION: SHORT SPREAD")
                lines.append(f"    SHORT {shares_a} shares of {pos.ticker_a} @ ${price_a:.2f} = ${shares_a * price_a:,.2f}")
                lines.append(f"    BUY  {shares_b} shares of {pos.ticker_b} @ ${price_b:.2f} = ${shares_b * price_b:,.2f}")

            lines.append(f"    Long:  ${long_dollars:,.2f}")
            lines.append(f"    Short: ${short_dollars:,.2f}")
            lines.append(f"    Net:   ${long_dollars - short_dollars:,.2f}")
            lines.append(f"    Entry Z: {pos.entry_z:+.2f}  |  Entry time: {pos.entry_time}")

            total_long_dollars += long_dollars
            total_short_dollars += short_dollars

    active = sum(1 for p in positions if p.signal != 0)
    gross = total_long_dollars + total_short_dollars
    net = total_long_dollars - total_short_dollars
    lines.append(f"\n{'='*70}")
    lines.append(f"  PORTFOLIO SUMMARY")
    lines.append(f"{'='*70}")
    lines.append(f"  Active pairs: {active}/{len(positions)}")
    lines.append(f"  Total long:   ${total_long_dollars:,.2f}")
    lines.append(f"  Total short:  ${total_short_dollars:,.2f}")
    lines.append(f"  Gross exposure: ${gross:,.2f} / $25,000 max")
    lines.append(f"  Net exposure:   ${net:,.2f} (target: $0)")
    lines.append(f"  Account size:   ${TOTAL_CAPITAL:,}")
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def log_signal(action: dict) -> None:
    """Append trade signal to signals CSV."""
    row = pd.DataFrame([action])
    if SIGNALS_FILE.exists():
        row.to_csv(SIGNALS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(SIGNALS_FILE, index=False)


def log(text: str) -> None:
    """Write to both console and the output log file."""
    print(text, flush=True)
    with open(OUTPUT_LOG, "a") as f:
        f.write(text + "\n")


def git_push(msg: str) -> None:
    """Commit and push signal/position data."""
    try:
        files = [
            "live_feed/signals.csv",
            "live_feed/positions.csv",
            "live_feed/trader_output.log",
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
    # Clear output log on startup
    with open(OUTPUT_LOG, "w") as f:
        f.write("")

    log("=" * 60)
    log("  LIVE STAT-ARB TRADER (5-min)")
    log(f"  Output log: {OUTPUT_LOG}")
    log("=" * 60)

    pair_configs = load_pairs()
    log(f"Loaded {len(pair_configs)} pairs.\n")

    # Initialize position trackers
    positions = []
    all_tickers = set()
    for p in pair_configs:
        pos = PairPosition(p["ticker_a"], p["ticker_b"], p["hedge_ratio"], p["sector"])
        positions.append(pos)
        all_tickers.update([p["ticker_a"], p["ticker_b"]])

    all_tickers = sorted(all_tickers)
    log(f"Tracking {len(all_tickers)} unique tickers across {len(positions)} pairs.\n")

    tick = 0
    while True:
        tick += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"[{now}] Tick #{tick}")

        # Fetch latest 5-min data
        try:
            prices = fetch_5min_data(all_tickers)
        except Exception as e:
            log(f"  Data fetch failed: {e}")
            time.sleep(INTERVAL_SECONDS)
            continue

        if prices.empty:
            log("  No data (market may be closed)")
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
                # Add share counts to the action
                price_a = prices[pos.ticker_a].iloc[-1] if pos.ticker_a in prices.columns else 0
                price_b = prices[pos.ticker_b].iloc[-1] if pos.ticker_b in prices.columns else 0
                shares_a = compute_shares(price_a, MAX_EXPOSURE_PER_PAIR)
                shares_b = compute_shares(price_b, MAX_EXPOSURE_PER_PAIR)
                action["shares_a"] = shares_a
                action["shares_b"] = shares_b
                action["price_a"] = price_a
                action["price_b"] = price_b

                actions.append(action)
                log_signal(action)

                if action["action"] == "EXIT":
                    log(f"  >>> EXIT: {action['pair']} @ z={z:+.2f}")
                    log(f"      Sell {shares_a} shares {pos.ticker_a}, Cover {shares_b} shares {pos.ticker_b}")
                else:
                    long_tk = action.get("long", "")
                    short_tk = action.get("short", "")
                    long_sh = shares_a if long_tk == pos.ticker_a else shares_b
                    short_sh = shares_b if short_tk == pos.ticker_b else shares_a
                    long_px = price_a if long_tk == pos.ticker_a else price_b
                    short_px = price_b if short_tk == pos.ticker_b else price_a
                    log(f"  >>> {action['action']}: {action['pair']} @ z={z:+.2f}")
                    log(f"      BUY  {long_sh} shares of {long_tk} @ ${long_px:.2f} = ${long_sh * long_px:,.2f}")
                    log(f"      SHORT {short_sh} shares of {short_tk} @ ${short_px:.2f} = ${short_sh * short_px:,.2f}")

        # Get latest prices for share calculations
        latest_prices = {}
        for t in all_tickers:
            if t in prices.columns:
                latest_prices[t] = prices[t].iloc[-1]

        # Print status table
        table = format_signal_table(positions, z_scores, latest_prices)
        log(table)

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
