#!/usr/bin/env python3
"""
Live 5-minute pairs trading + dynamic portfolio construction (updates every 1 minute).

MATCHES YOUR PROMPT:
- Account Size: $10,000
- Max Gross Exposure: $5,000
- Target Net: ~0 (dollar-neutral per pair)
- Strategy: cointegration pairs, z-score mean reversion
- Entry: |z| >= 2.0
- Exit:  |z| <= 0.5
- Spread computed on 5-minute closes (Yahoo interval="5m")
- Recompute / rebalance every 1 minute
- Expand universe (up to 25 pairs from active_pairs.csv)
- Dynamic ranking each minute:
    score = |z| + quality boosts - volatility penalty (lightweight)
- Activate only strongest signals (max 5 active pairs)
- Constraints:
    * max 2 pairs per sector active
    * max 2 active pairs per ticker (overlap control)
- Dynamic sizing:
    * gross per pair varies with score
    * total gross across active pairs <= $5,000
- Risk controls:
    * stop on |z| >= 3.5
    * stop on per-pair PnL <= -$500 (5% of account)
- Rotation control:
    * don’t churn: minimum hold time + “new must be 20% stronger than weakest”
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -----------------------------
# Runtime / files
# -----------------------------
INTERVAL_SECONDS = 60  # update every minute
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNALS_FILE = PROJECT_ROOT / "live_feed" / "signals.csv"
POSITIONS_FILE = PROJECT_ROOT / "live_feed" / "positions.csv"
OUTPUT_LOG = PROJECT_ROOT / "live_feed" / "trader_output.log"

# -----------------------------
# Strategy + portfolio config
# -----------------------------
# Spread + z-score
ZSCORE_LOOKBACK = 60       # 60 x 5min = ~5 hours
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
ZSCORE_STOP = 3.5

# Account / exposure
TOTAL_CAPITAL = 10_000
MAX_GROSS_EXPOSURE = 5_000     # per prompt
MAX_ACTIVE_PAIRS = 5
MAX_PAIRS = 25

# Constraints
MAX_PAIRS_PER_SECTOR = 2
MAX_ACTIVE_PER_TICKER = 2

# Risk cap per pair (5% of account)
MAX_LOSS_PER_PAIR = 0.05 * TOTAL_CAPITAL  # $500

# Rotation / overtrading controls
MIN_HOLD_SECONDS = 10 * 60
ROTATION_MARGIN = 1.20  # new candidate score must be 20% better than weakest active
PREENTRY_BUFFER = 0.25  # start "watching" a little before entry to reduce idle time

# Output noise control
WATCHLIST_THRESHOLD = 1.0  # only print flat pairs with |z| >= this

# Sizing floors/ceilings for $10k account
MIN_GROSS_PER_PAIR = 600.0
MAX_GROSS_PER_PAIR = 2_000.0


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class PairConfig:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    sector: str

    @property
    def label(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"


class PairPosition:
    """Track state + fill details for one pair."""

    def __init__(self, cfg: PairConfig):
        self.cfg = cfg

        # 0 flat, +1 long spread (buy A, short B), -1 short spread (short A, buy B)
        self.signal: int = 0

        # entry context
        self.entry_z: Optional[float] = None
        self.entry_time: Optional[str] = None
        self.entry_ts_epoch: Optional[float] = None

        # sizing context (shares)
        self.qty_a: int = 0
        self.qty_b: int = 0
        self.entry_price_a: float = 0.0
        self.entry_price_b: float = 0.0
        self.allocated_gross: float = 0.0

        # last mtm
        self.last_pnl: float = 0.0

    def is_active(self) -> bool:
        return self.signal != 0

    def tickers(self) -> Tuple[str, str]:
        return (self.cfg.ticker_a, self.cfg.ticker_b)

    def update_mtm(self, price_a: float, price_b: float) -> float:
        """Mark-to-market PnL estimate for risk control."""
        if self.signal == 0 or self.qty_a <= 0 or self.qty_b <= 0:
            self.last_pnl = 0.0
            return 0.0

        if self.signal == 1:
            # LONG spread: +A, -B
            pnl_a = (price_a - self.entry_price_a) * self.qty_a
            pnl_b = (self.entry_price_b - price_b) * self.qty_b
            self.last_pnl = pnl_a + pnl_b
        else:
            # SHORT spread: -A, +B
            pnl_a = (self.entry_price_a - price_a) * self.qty_a
            pnl_b = (price_b - self.entry_price_b) * self.qty_b
            self.last_pnl = pnl_a + pnl_b

        return self.last_pnl

    def can_rotate_out(self) -> bool:
        if self.entry_ts_epoch is None:
            return True
        return (time.time() - self.entry_ts_epoch) >= MIN_HOLD_SECONDS

    def enter(
        self,
        z: float,
        now_str: str,
        qty_a: int,
        qty_b: int,
        price_a: float,
        price_b: float,
        gross: float,
        side: int,
    ) -> Dict:
        """Open a position with explicit sizing."""
        self.signal = side
        self.entry_z = z
        self.entry_time = now_str
        self.entry_ts_epoch = time.time()

        self.qty_a = qty_a
        self.qty_b = qty_b
        self.entry_price_a = price_a
        self.entry_price_b = price_b
        self.allocated_gross = gross
        self.last_pnl = 0.0

        if side == 1:
            action = {
                "action": "ENTER_LONG_SPREAD",
                "long": self.cfg.ticker_a,
                "short": self.cfg.ticker_b,
            }
        else:
            action = {
                "action": "ENTER_SHORT_SPREAD",
                "long": self.cfg.ticker_b,
                "short": self.cfg.ticker_a,
            }

        action.update({
            "pair": self.cfg.label,
            "hedge_ratio": self.cfg.hedge_ratio,
            "z_score": z,
            "timestamp": now_str,
            "sector": self.cfg.sector,
            "qty_a": qty_a,
            "qty_b": qty_b,
            "price_a": price_a,
            "price_b": price_b,
            "gross_alloc": gross,
        })
        return action

    def exit(self, z: float, now_str: str, reason: str) -> Dict:
        """Close a position (state reset)."""
        prev = "LONG_SPREAD" if self.signal == 1 else "SHORT_SPREAD"
        action = {
            "action": "EXIT",
            "prev_signal": prev,
            "entry_z": self.entry_z,
            "exit_z": z,
            "timestamp": now_str,
            "pair": self.cfg.label,
            "hedge_ratio": self.cfg.hedge_ratio,
            "z_score": z,
            "sector": self.cfg.sector,
            "qty_a": self.qty_a,
            "qty_b": self.qty_b,
            "entry_time": self.entry_time,
            "exit_reason": reason,
            "mtm_pnl": self.last_pnl,
            "gross_alloc": self.allocated_gross,
        }

        # reset
        self.signal = 0
        self.entry_z = None
        self.entry_time = None
        self.entry_ts_epoch = None
        self.qty_a = 0
        self.qty_b = 0
        self.entry_price_a = 0.0
        self.entry_price_b = 0.0
        self.allocated_gross = 0.0
        self.last_pnl = 0.0

        return action


# -----------------------------
# I/O + helpers
# -----------------------------
def log(text: str) -> None:
    print(text, flush=True)
    with open(OUTPUT_LOG, "a") as f:
        f.write(text + "\n")


def log_signal(action: Dict) -> None:
    row = pd.DataFrame([action])
    if SIGNALS_FILE.exists():
        row.to_csv(SIGNALS_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(SIGNALS_FILE, index=False)


def git_push(msg: str) -> None:
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


def load_pairs() -> List[PairConfig]:
    pairs_csv = PROJECT_ROOT / "live_feed" / "active_pairs.csv"

    if pairs_csv.exists():
        df = pd.read_csv(pairs_csv).head(MAX_PAIRS)
        out: List[PairConfig] = []
        for _, r in df.iterrows():
            out.append(PairConfig(
                ticker_a=str(r["ticker_a"]).strip(),
                ticker_b=str(r["ticker_b"]).strip(),
                hedge_ratio=float(r["hedge_ratio"]),
                sector=str(r.get("sector", "Unknown")).strip(),
            ))
        return out

    log("WARNING: No live_feed/active_pairs.csv found. Using fallback pairs.")
    return [
        PairConfig("KO", "PEP", 1.0, "Consumer Staples"),
        PairConfig("XOM", "CVX", 1.0, "Energy"),
        PairConfig("GS", "MS", 1.0, "Financials"),
        PairConfig("JPM", "BAC", 1.0, "Financials"),
        PairConfig("HD", "LOW", 1.0, "Consumer Discretionary"),
    ]


def fetch_5min_data(tickers: List[str]) -> pd.DataFrame:
    df = yf.download(
        tickers,
        period="5d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna()


def compute_zscore(spread: pd.Series, lookback: int = ZSCORE_LOOKBACK) -> float:
    if len(spread) < 5:
        return 0.0
    lb = min(lookback, len(spread))
    recent = spread.iloc[-lb:]
    std = float(recent.std())
    if std < 1e-8:
        return 0.0
    mean = float(recent.mean())
    return float((spread.iloc[-1] - mean) / std)


def compute_score(z: float) -> float:
    """
    Lightweight ranking score.
    You asked for: |z| + stability + liquidity - vol penalty.
    With yfinance alone we don't have true liquidity/half-life each minute,
    so we do a practical proxy:

    score = |z| + small boost for stronger signals past entry
            - small penalty for extremely large z (often indicates regime break)
    """
    az = abs(z)
    if az < (ZSCORE_ENTRY - PREENTRY_BUFFER):
        return 0.0
    boost = 0.15 * max(0.0, az - ZSCORE_ENTRY)  # stronger beyond entry gets more weight
    penalty = 0.10 * max(0.0, az - 3.0)         # discourage very extreme tail
    return max(0.0, az + boost - penalty)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def alloc_gross_by_scores(selected: List[PairConfig], score_map: Dict[str, float]) -> Dict[str, float]:
    if not selected:
        return {}

    weights = []
    labels = []
    for p in selected:
        labels.append(p.label)
        weights.append(max(score_map.get(p.label, 0.1), 0.1))

    s = sum(weights) or 1.0
    alloc = {lab: MAX_GROSS_EXPOSURE * (w / s) for lab, w in zip(labels, weights)}

    # Apply per-pair min/max + renormalize if needed
    for lab in alloc:
        alloc[lab] = clamp(alloc[lab], MIN_GROSS_PER_PAIR, MAX_GROSS_PER_PAIR)

    total = sum(alloc.values())
    if total > MAX_GROSS_EXPOSURE:
        scale = MAX_GROSS_EXPOSURE / total
        for lab in alloc:
            alloc[lab] *= scale

    return alloc


def shares_for_pair(
    cfg: PairConfig,
    gross_alloc: float,
    price_a: float,
    price_b: float,
) -> Tuple[int, int]:
    """
    Dollar-neutral-ish sizing:
      dollars_leg = gross/2
      qty_a = dollars_leg / price_a
      qty_b = (dollars_leg / price_b) * |hedge_ratio|
    """
    dollars_leg = gross_alloc / 2.0
    qty_a = int(dollars_leg / max(price_a, 1e-9))
    qty_b = int((dollars_leg / max(price_b, 1e-9)) * abs(cfg.hedge_ratio))

    qty_a = max(qty_a, 1)
    qty_b = max(qty_b, 1)
    return qty_a, qty_b


def sector_counts(active_positions: List[PairPosition]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for p in active_positions:
        if p.is_active():
            c[p.cfg.sector] = c.get(p.cfg.sector, 0) + 1
    return c


def ticker_counts(active_positions: List[PairPosition]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for p in active_positions:
        if p.is_active():
            a, b = p.tickers()
            c[a] = c.get(a, 0) + 1
            c[b] = c.get(b, 0) + 1
    return c


def weakest_active(active_positions: List[PairPosition], score_map: Dict[str, float]) -> Optional[PairPosition]:
    weakest = None
    weakest_score = float("inf")
    for p in active_positions:
        if not p.is_active():
            continue
        sc = score_map.get(p.cfg.label, 0.0)
        if sc < weakest_score:
            weakest_score = sc
            weakest = p
    return weakest


# -----------------------------
# Formatting
# -----------------------------
def format_signal_table_dynamic(
    positions: list[PairPosition],
    z_scores: dict,
    latest_prices: dict,
    allocs: dict,
    targets: list[str],
    total_capital: float,
    max_gross: float,
) -> str:
    """
    Old-style output:
    - Shows each pair with Z, prices
    - Shows FLAT/watch/active status
    - Shows "If z hits..." projected shares based on the pair's current allocation
    - Summarizes portfolio gross/net
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("\n" + "=" * 60)
    lines.append("  LIVE STAT-ARB TRADER (5-min) — Dynamic Portfolio")
    lines.append(f"  Output log: {OUTPUT_LOG}")
    lines.append("=" * 60)

    lines.append(f"\nLoaded {len(positions)} pairs.\n")

    # Count tickers for display
    tickers = set()
    for p in positions:
        tickers.add(p.cfg.ticker_a)
        tickers.add(p.cfg.ticker_b)
    lines.append(f"Tracking {len(tickers)} unique tickers across {len(positions)} pairs.\n")

    lines.append("\n" + "=" * 70)
    lines.append(f"  LIVE SIGNALS — {now}")
    lines.append("=" * 70)

    total_long_dollars = 0.0
    total_short_dollars = 0.0

    hidden = 0
    for pos in positions:
        label = pos.cfg.label
        z = z_scores.get(label, 0.0)
        price_a = latest_prices.get(pos.cfg.ticker_a, 0.0)
        price_b = latest_prices.get(pos.cfg.ticker_b, 0.0)

        # Hide quiet flat pairs unless they're targets
        if pos.signal == 0 and abs(z) < WATCHLIST_THRESHOLD and label not in targets:
            hidden += 1
            continue

        lines.append(f"\n  PAIR: {pos.cfg.ticker_a} / {pos.cfg.ticker_b}  ({pos.cfg.sector})")
        lines.append(f"  Z-Score: {z:+.2f}")
        lines.append(f"  Prices: {pos.cfg.ticker_a} = ${price_a:.2f}  |  {pos.cfg.ticker_b} = ${price_b:.2f}")

        # Decide what gross to preview with
        gross_alloc = float(allocs.get(label, MIN_GROSS_PER_PAIR))
        dollars_leg = gross_alloc / 2.0

        # Compute preview shares using hedge ratio sizing
        shares_a = max(1, int(dollars_leg / max(price_a, 1e-9)))
        shares_b = max(1, int((dollars_leg / max(price_b, 1e-9)) * abs(pos.cfg.hedge_ratio)))

        if pos.signal == 0:
            # Status label
            if abs(z) < ZSCORE_ENTRY:
                if label in targets:
                    status = "FLAT — watching (ranked target)"
                else:
                    status = "FLAT — no trade (z within normal range)"
            else:
                status = "FLAT — entry triggered (eligible)"

            lines.append(f"  Status: {status}")

            # Old-style “If z hits …” preview
            if z >= 0:
                lines.append(
                    f"  If z hits +{ZSCORE_ENTRY:.1f} → Short {shares_a} shares {pos.cfg.ticker_a} "
                    f"(${shares_a * price_a:,.2f})"
                )
                lines.append(
                    f"                    Buy {shares_b} shares {pos.cfg.ticker_b} "
                    f"(${shares_b * price_b:,.2f})"
                )
            else:
                lines.append(
                    f"  If z hits -{ZSCORE_ENTRY:.1f} → Buy {shares_a} shares {pos.cfg.ticker_a} "
                    f"(${shares_a * price_a:,.2f})"
                )
                lines.append(
                    f"                    Short {shares_b} shares {pos.cfg.ticker_b} "
                    f"(${shares_b * price_b:,.2f})"
                )

        else:
            # Active trade display (uses actual entry sizes saved on pos)
            if pos.signal == 1:
                lines.append("  Status: ACTIVE — LONG SPREAD")
                long_dollars = pos.qty_a * price_a
                short_dollars = pos.qty_b * price_b
                lines.append(
                    f"    BUY  {pos.qty_a} shares of {pos.cfg.ticker_a} @ ${price_a:.2f} "
                    f"= ${pos.qty_a * price_a:,.2f}"
                )
                lines.append(
                    f"    SHORT {pos.qty_b} shares of {pos.cfg.ticker_b} @ ${price_b:.2f} "
                    f"= ${pos.qty_b * price_b:,.2f}"
                )
            else:
                lines.append("  Status: ACTIVE — SHORT SPREAD")
                long_dollars = pos.qty_b * price_b
                short_dollars = pos.qty_a * price_a
                lines.append(
                    f"    SHORT {pos.qty_a} shares of {pos.cfg.ticker_a} @ ${price_a:.2f} "
                    f"= ${pos.qty_a * price_a:,.2f}"
                )
                lines.append(
                    f"    BUY  {pos.qty_b} shares of {pos.cfg.ticker_b} @ ${price_b:.2f} "
                    f"= ${pos.qty_b * price_b:,.2f}"
                )

            lines.append(f"    Long:  ${long_dollars:,.2f}")
            lines.append(f"    Short: ${short_dollars:,.2f}")
            lines.append(f"    Net:   ${long_dollars - short_dollars:,.2f}")
            lines.append(f"    Entry Z: {pos.entry_z:+.2f}  |  Entry time: {pos.entry_time}")
            lines.append(f"    MTM PnL: ${pos.last_pnl:,.2f}")

            total_long_dollars += long_dollars
            total_short_dollars += short_dollars

    active = sum(1 for p in positions if p.signal != 0)
    gross = total_long_dollars + total_short_dollars
    net = total_long_dollars - total_short_dollars
    watching = len(positions) - active - hidden

    lines.append("\n" + "=" * 70)
    lines.append("  PORTFOLIO SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Active pairs: {active}/{len(positions)}")
    lines.append(f"  Total long:   ${total_long_dollars:,.2f}")
    lines.append(f"  Total short:  ${total_short_dollars:,.2f}")
    lines.append(f"  Gross exposure: ${gross:,.2f} / ${max_gross:,.0f} max")
    lines.append(f"  Net exposure:   ${net:,.2f} (target: $0)")
    lines.append(f"  Account size:   ${total_capital:,.0f}")
    lines.append("=" * 70 + "\n")

    return "\n".join(lines)


# -----------------------------
# Main loop
# -----------------------------
def run_trader() -> None:
    # Clear output log on startup
    with open(OUTPUT_LOG, "w") as f:
        f.write("")

    log("=" * 60)
    log("  LIVE STAT-ARB TRADER (5-min) — Dynamic Portfolio ($10k)")
    log(f"  Output log: {OUTPUT_LOG}")
    log("=" * 60)

    pair_cfgs = load_pairs()
    log(f"Loaded {len(pair_cfgs)} pairs.\n")

    # Initialize positions
    positions: List[PairPosition] = []
    all_tickers = set()
    for cfg in pair_cfgs:
        positions.append(PairPosition(cfg))
        all_tickers.update([cfg.ticker_a, cfg.ticker_b])

    all_tickers = sorted(all_tickers)
    log(f"Tracking {len(all_tickers)} unique tickers across {len(positions)} pairs.\n")

    tick = 0
    while True:
        tick += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"[{now_str}] Tick #{tick}")

        # 1) Fetch latest 5-min data
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

        # Latest prices snapshot
        latest_prices: Dict[str, float] = {}
        for t in all_tickers:
            if t in prices.columns:
                latest_prices[t] = float(prices[t].iloc[-1])

        # 2) Compute z-scores + scores, update MTM PnL for active positions
        z_scores: Dict[str, float] = {}
        score_map: Dict[str, float] = {}

        for pos in positions:
            a, b = pos.cfg.ticker_a, pos.cfg.ticker_b
            if a not in prices.columns or b not in prices.columns:
                continue

            spread = prices[a] - pos.cfg.hedge_ratio * prices[b]
            z = compute_zscore(spread)
            z_scores[pos.cfg.label] = z
            score_map[pos.cfg.label] = compute_score(z)

            # update mtm for risk controls
            pa = latest_prices.get(a, 0.0)
            pb = latest_prices.get(b, 0.0)
            pos.update_mtm(pa, pb)

        actions: List[Dict] = []

        # 3) Exit / stop logic first (risk control always wins)
        for pos in positions:
            if not pos.is_active():
                continue

            label = pos.cfg.label
            z = z_scores.get(label, 0.0)

            # Stop on extreme z
            if abs(z) >= ZSCORE_STOP:
                act = pos.exit(z, now_str, reason=f"Z_STOP(|z|>={ZSCORE_STOP})")
                actions.append(act)
                log_signal(act)
                log(f"  >>> EXIT (Z-STOP): {label} @ z={z:+.2f}")
                continue

            # Stop on max loss
            if pos.last_pnl <= -MAX_LOSS_PER_PAIR:
                act = pos.exit(z, now_str, reason=f"LOSS_STOP(PnL<={-MAX_LOSS_PER_PAIR:.0f})")
                actions.append(act)
                log_signal(act)
                log(f"  >>> EXIT (LOSS-STOP): {label} PnL=${pos.last_pnl:,.2f} @ z={z:+.2f}")
                continue

            # Normal exit
            if abs(z) <= ZSCORE_EXIT:
                act = pos.exit(z, now_str, reason=f"EXIT(|z|<={ZSCORE_EXIT})")
                actions.append(act)
                log_signal(act)
                log(f"  >>> EXIT: {label} @ z={z:+.2f}")
                continue

        # 4) Determine active + eligible candidates (rank)
        active_positions = [p for p in positions if p.is_active()]
        active_labels = {p.cfg.label for p in active_positions}

        # Rank by score (desc)
        ranked = sorted(
            ((p, score_map.get(p.cfg.label, 0.0)) for p in positions),
            key=lambda x: x[1],
            reverse=True,
        )

        # Candidate list: score>0 and not active
        candidates: List[PairConfig] = []
        for p, sc in ranked:
            if sc <= 0:
                continue
            if p.cfg.label in active_labels:
                continue
            candidates.append(p.cfg)

        # 5) Build target set: keep current active + add best candidates until MAX_ACTIVE_PAIRS
        targets: List[PairConfig] = [p.cfg for p in active_positions]  # keep currently active
        sector_ct = sector_counts(active_positions)
        ticker_ct = ticker_counts(active_positions)

        # Fill slots with highest-ranked candidates subject to constraints
        for cfg in candidates:
            if len(targets) >= MAX_ACTIVE_PAIRS:
                break

            # sector limit
            if sector_ct.get(cfg.sector, 0) >= MAX_PAIRS_PER_SECTOR:
                continue

            # ticker overlap limit
            if ticker_ct.get(cfg.ticker_a, 0) >= MAX_ACTIVE_PER_TICKER:
                continue
            if ticker_ct.get(cfg.ticker_b, 0) >= MAX_ACTIVE_PER_TICKER:
                continue

            targets.append(cfg)
            sector_ct[cfg.sector] = sector_ct.get(cfg.sector, 0) + 1
            ticker_ct[cfg.ticker_a] = ticker_ct.get(cfg.ticker_a, 0) + 1
            ticker_ct[cfg.ticker_b] = ticker_ct.get(cfg.ticker_b, 0) + 1

        # Allocate gross across targets (dynamic sizing)
        allocs = alloc_gross_by_scores(targets, score_map)

        # 6) Rotation gate: if no open slots, only replace weakest if candidate is materially stronger
        # (We implement rotation only at ENTRY time to reduce churn)
        weak = weakest_active(active_positions, score_map)
        weak_score = score_map.get(weak.cfg.label, 0.0) if weak else 0.0

        # 7) Enter new positions if entry is triggered and constraints allow
        #    IMPORTANT: We only enter for pairs in targets, and only if |z| >= ENTRY.
        selected_targets_labels = {cfg.label for cfg in targets}

        for pos in positions:
            if len(active_positions) >= MAX_ACTIVE_PAIRS:
                break
            if pos.is_active():
                continue

            label = pos.cfg.label
            if label not in selected_targets_labels:
                continue

            z = z_scores.get(label, 0.0)
            if abs(z) < ZSCORE_ENTRY:
                continue

            sc = score_map.get(label, 0.0)

            # If slots full, attempt rotate out weakest (only if weakest can rotate)
            if len(active_positions) >= MAX_ACTIVE_PAIRS:
                if not weak or not weak.can_rotate_out():
                    continue
                if sc < weak_score * ROTATION_MARGIN:
                    continue

                # rotate out weakest
                z_w = z_scores.get(weak.cfg.label, 0.0)
                act = weak.exit(z_w, now_str, reason=f"ROTATE_OUT(replaced_by={label})")
                actions.append(act)
                log_signal(act)
                log(f"  >>> ROTATE OUT: {weak.cfg.label} (weak) replaced by {label}")

                # refresh active list
                active_positions = [p for p in positions if p.is_active()]
                active_labels = {p.cfg.label for p in active_positions}

            # Sizing
            pa = latest_prices.get(pos.cfg.ticker_a, 0.0)
            pb = latest_prices.get(pos.cfg.ticker_b, 0.0)
            gross = float(allocs.get(label, MIN_GROSS_PER_PAIR))
            qty_a, qty_b = shares_for_pair(pos.cfg, gross, pa, pb)

            # side decision
            side = -1 if z >= 0 else 1  # +z => SHORT spread; -z => LONG spread

            act = pos.enter(
                z=z,
                now_str=now_str,
                qty_a=qty_a,
                qty_b=qty_b,
                price_a=pa,
                price_b=pb,
                gross=gross,
                side=side,
            )
            actions.append(act)
            log_signal(act)

            if side == 1:
                log(f"  >>> ENTER LONG SPREAD: {label} @ z={z:+.2f}")
                log(f"      BUY  {qty_a} {pos.cfg.ticker_a} @ ${pa:.2f} = ${qty_a * pa:,.2f}")
                log(f"      SHORT {qty_b} {pos.cfg.ticker_b} @ ${pb:.2f} = ${qty_b * pb:,.2f}")
            else:
                log(f"  >>> ENTER SHORT SPREAD: {label} @ z={z:+.2f}")
                log(f"      SHORT {qty_a} {pos.cfg.ticker_a} @ ${pa:.2f} = ${qty_a * pa:,.2f}")
                log(f"      BUY  {qty_b} {pos.cfg.ticker_b} @ ${pb:.2f} = ${qty_b * pb:,.2f}")

            active_positions = [p for p in positions if p.is_active()]
            weak = weakest_active(active_positions, score_map)
            weak_score = score_map.get(weak.cfg.label, 0.0) if weak else 0.0

        # 8) Print status table
        target_labels_print = [cfg.label for cfg in targets]
        table = format_signal_table(positions, z_scores, latest_prices, score_map, target_labels_print)
        log(table)

        # 9) Save current positions snapshot
        pos_rows = []
        for pos in positions:
            label = pos.cfg.label
            pos_rows.append({
                "pair": label,
                "sector": pos.cfg.sector,
                "hedge_ratio": pos.cfg.hedge_ratio,
                "signal": pos.signal,
                "z_score": z_scores.get(label, 0.0),
                "score": score_map.get(label, 0.0),
                "qty_a": pos.qty_a,
                "qty_b": pos.qty_b,
                "entry_time": pos.entry_time,
                "entry_z": pos.entry_z,
                "allocated_gross": pos.allocated_gross,
                "mtm_pnl": pos.last_pnl,
                "ticker_a": pos.cfg.ticker_a,
                "ticker_b": pos.cfg.ticker_b,
            })
        pd.DataFrame(pos_rows).to_csv(POSITIONS_FILE, index=False)

        # 10) Push to GitHub
        if actions:
            git_push(f"trade actions tick #{tick} — {now_str}")
        else:
            git_push(f"position update #{tick} — {now_str}")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_trader()
