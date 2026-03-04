# Equity Statistical Arbitrage

Market-neutral pairs-trading system using cointegration and z-score mean reversion on U.S. equities. Includes a walk-forward backtester and a live intraday paper trading engine powered by Alpaca.

## Strategy

1. **Pair discovery** — Scan ~900 NASDAQ/NYSE tickers (S&P 500 + S&P 400) for intra-sector Engle-Granger cointegration (p < 0.02), ADF stationarity (p < 0.02), R² > 0.60, and mean-reversion half-life < 30 bars.
2. **Hedge ratio** — OLS regression slope between the two price series.
3. **Signal generation** — Compute the rolling z-score of the spread (60-bar lookback). Enter at |z| >= 2.0, exit at |z| <= 0.5.
4. **Risk controls** — Hard stop at |z| >= 3.25, trailing stop (2% of entry cost from peak P&L), time stop after 1 trading day, 1-day cooldown after forced exits. Gross exposure capped at $100k. Sector concentration limits.
5. **Position sizing** — Vol-adjusted dollar-neutral exposure ($10k base per leg, scaled inversely to spread volatility). Loss streak detection scales down after 3 consecutive losses.
6. **Intraday only** — 30-min opening cooldown, no new entries after 3:30 PM ET, force exit all positions at 3:45 PM ET. No overnight holds.
7. **Order execution** — Limit orders with 0.05% buffer, 30-second timeout, automatic market order fallback. Actual fill prices tracked for P&L.
8. **Backtest** — Walk-forward: train on 252 days, trade on 21 days (~monthly pair rediscovery), roll forward.

## Quick Start

### Backtest
```bash
pip install -r requirements.txt
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
python main.py
```

### Live Paper Trading
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python -m live_feed.trader
```

### Scan for New Pairs
```bash
python -m strategy.scanner
```
Runs automatically every Sunday at 8 PM ET.

## Project Structure

```
stat_arb/
├── CLAUDE.md                  # Project context for Claude Code sessions
├── config.py                  # Tickers, thresholds, risk params
├── main.py                    # Backtest entry point
├── requirements.txt
├── data/
│   ├── provider.py            # AlpacaProvider (daily bars for backtest)
│   ├── universe.py            # S&P 500 + 400 ticker fetcher
│   └── sectors.py             # GICS sector classification
├── strategy/
│   ├── pairs.py               # Cointegration scan, z-score signals
│   └── scanner.py             # Full NASDAQ/NYSE pair scanner
├── portfolio/
│   └── construction.py        # Dollar-neutral weights, risk limits
├── backtest/
│   └── engine.py              # Walk-forward backtest loop
├── metrics/
│   └── performance.py         # Sharpe, drawdown, turnover, beta
└── live_feed/
    ├── alpaca_client.py        # Alpaca API: data, orders (limit + market), account
    ├── trader.py               # Live trading engine (1-min ticks, 5-min bars)
    ├── fetcher.py              # Price feed for broad universe
    ├── active_pairs.csv        # Current pairs being traded (8 pairs)
    ├── position_state.json     # Persisted position state for restart recovery
    ├── positions.csv           # Live position state
    ├── signals.csv             # Trade signal log
    ├── slippage.csv            # Fill price vs signal price tracking
    └── pair_pnl.csv            # Per-pair consecutive loss counts
```

## Live Trader Parameters

| Parameter | Value | Description |
|---|---|---|
| `INTERVAL_SECONDS` | 60 | Tick frequency (1 minute) |
| `TOTAL_CAPITAL` | 100,000 | Paper account size |
| `BASE_EXPOSURE_PER_PAIR` | 10,000 | Dollar amount per leg (vol-adjusted) |
| `ZSCORE_ENTRY` | 2.0 | Z-score threshold to enter a trade |
| `ZSCORE_EXIT` | 0.5 | Z-score threshold to exit (mean reversion) |
| `ZSCORE_HARD_STOP` | 3.25 | Force exit if z-score blows out |
| `TRAILING_STOP_PCT` | 0.02 | Exit if P&L drops 2% of entry cost from peak |
| `TIME_STOP_BARS` | 78 | Force exit after 1 trading day (6.5hrs at 5-min) |
| `COOLDOWN_BARS` | 78 | 1-day lockout after hard/time/trailing stop |
| `OPEN_COOLDOWN_MINUTES` | 30 | No entries for first 30 min after market open |
| `EOD_NO_ENTRY_TIME` | 3:30 PM ET | Block new entries |
| `EOD_CLOSE_TIME` | 3:45 PM ET | Force exit all positions |
| `MAX_GROSS_EXPOSURE` | 100,000 | Cap total gross exposure |
| `MAX_SECTOR_ACTIVE` | 3 | Max active positions per sector |
| `LOSS_STREAK_CUTOFF` | 3 | Reduce size after 3 consecutive losses |
| `LIMIT_ORDER_BUFFER_PCT` | 0.05% | Limit price cushion |
| `LIMIT_ORDER_TIMEOUT_SECS` | 30 | Timeout before market fallback |

## Pair Scanner Parameters

| Parameter | Value | Description |
|---|---|---|
| Cointegration p-value | < 0.02 | Engle-Granger test |
| ADF p-value | < 0.02 | Spread stationarity |
| R² | > 0.60 | Regression fit quality |
| Half-life | < 30 bars | Mean reversion speed |
| Max pairs | 8 | Top pairs selected |
| Max per sector | 2 | Diversification constraint |
| Lookback | 10 days | Data window for pair evaluation |

## Telegram Monitoring

The algo sends real-time alerts to Telegram:

| Alert | Timing | Content |
|---|---|---|
| **Entry** | On trade open | Pair, shares, dollar amounts per leg, z-score, imbalance warning |
| **Exit** | On trade close | Pair, reason, bars held, P&L from actual Alpaca fill prices |
| **Morning check** | 10:00 AM ET | Account equity, positions, Alpaca vs internal state reconciliation |
| **Hourly pulse** | Every hour | Active count, unrealized P&L, best/worst pair, today's realized |
| **Daily recap** | 4:00 PM ET | Total P&L, win rate, per-pair breakdown, exit types, slippage, diagnostics |
| **Weekly report** | Friday 4:10 PM ET | Week's trade stats, per-pair performance, slippage, pair health |

## Data

- **Backtest**: Alpaca daily bars via `AlpacaProvider`
- **Live trading**: Alpaca 5-min bars via IEX feed (free real-time)
- **Order execution**: Limit orders (0.05% buffer) with 30s timeout, automatic market order fallback. Fill prices captured for accurate P&L.

## Environment Variables

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
export TELEGRAM_BOT_TOKEN="your_bot_token"  # optional, for alerts
export TELEGRAM_CHAT_ID="your_chat_id"      # optional, for alerts
```

## Performance Metrics

- Annualized return & volatility
- Sharpe ratio
- Maximum drawdown
- Annualized turnover
- Portfolio beta vs SPY
