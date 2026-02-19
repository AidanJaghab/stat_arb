# Equity Statistical Arbitrage

Market-neutral pairs-trading system using cointegration and z-score mean reversion on U.S. equities. Includes a walk-forward backtester and a live paper trading engine powered by Alpaca.

## Strategy

1. **Pair discovery** — Scan ~900 NASDAQ/NYSE tickers (S&P 500 + S&P 400) for intra-sector Engle-Granger cointegration (p < 0.05), ADF stationarity, and mean-reversion half-life.
2. **Hedge ratio** — OLS regression slope between the two price series.
3. **Signal generation** — Compute the rolling z-score of the spread. Enter at |z| >= 2, exit at |z| <= 0.5.
4. **Risk controls** — Hard stop at |z| >= 3.25, time stop after 5 trading days, 1-day cooldown after forced exits.
5. **Portfolio construction** — Dollar-neutral weights with per-position caps (5%) and gross leverage limit (4x).
6. **Backtest** — Walk-forward: train on 252 days, trade on 21 days (~monthly pair rediscovery), roll forward.
7. **Live trading** — Paper trading via Alpaca with automatic order execution, position reconciliation on startup, and retry logic for exits.

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
python -m live_feed.trader
```

### Scan for New Pairs
```bash
python -m strategy.scanner
```

## Project Structure

```
stat_arb/
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
    ├── alpaca_client.py        # Alpaca API: data, orders, account
    ├── trader.py               # Live trading engine (runs every 60s)
    ├── fetcher.py              # Price feed for broad universe
    ├── active_pairs.csv        # Current pairs being traded
    └── positions.csv           # Live position state
```

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | ~900 S&P 500 + 400 names | Trading universe |
| `COINT_PVALUE` | 0.05 | Max p-value for cointegration |
| `ZSCORE_ENTRY` | 2.0 | Z-score threshold to enter a trade |
| `ZSCORE_EXIT` | 0.5 | Z-score threshold to exit |
| `TRAINING_WINDOW` | 252 | Training lookback (trading days) |
| `TRADING_WINDOW` | 21 | Out-of-sample trading period (~1 month) |
| `MAX_POSITION_WEIGHT` | 0.05 | Max weight per position leg |
| `MAX_GROSS_LEVERAGE` | 4.0 | Max sum of absolute weights |

### Live Trader Parameters (`live_feed/trader.py`)

| Parameter | Default | Description |
|---|---|---|
| `INTERVAL_SECONDS` | 60 | Tick frequency |
| `TOTAL_CAPITAL` | 100,000 | Paper account size |
| `MAX_EXPOSURE_PER_PAIR` | 5,000 | Dollar amount per leg |
| `ZSCORE_HARD_STOP` | 3.25 | Force exit if z-score blows out |
| `TIME_STOP_BARS` | 390 | Force exit after 5 trading days |
| `COOLDOWN_BARS` | 78 | 1-day lockout after forced exit |

## Data

- **Backtest**: Alpaca daily bars via `AlpacaProvider` (15-min delayed SIP, free tier)
- **Live trading**: Alpaca 5-min bars via SIP feed (15-min delayed, free tier)
- Market orders for execution (ensures both legs fill simultaneously)

## Environment Variables

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

## Performance Metrics

- Annualized return & volatility
- Sharpe ratio
- Maximum drawdown
- Annualized turnover
- Portfolio beta vs SPY
