# Equity Statistical Arbitrage

Market-neutral pairs-trading system using cointegration and z-score mean reversion on U.S. equities.

## Strategy

1. **Pair discovery** — Scan all unique pairs in the universe for Engle-Granger cointegration (p < 0.05).
2. **Hedge ratio** — OLS regression slope between the two price series.
3. **Signal generation** — Compute the rolling z-score of the spread. Enter at |z| >= 2, exit at |z| <= 0.5.
4. **Portfolio construction** — Dollar-neutral weights with per-position caps (5%) and gross leverage limit (4x).
5. **Backtest** — Walk-forward: train on 252 days, trade on 63 days, roll forward.

## Quick Start

```bash
cd stat_arb
pip install -r requirements.txt
python main.py
```

Output: performance summary table printed to console + `equity_curve.png` saved to the project root.

## Project Structure

```
stat_arb/
├── config.py               # Tickers, thresholds, risk params
├── main.py                 # End-to-end entry point
├── data/
│   └── provider.py         # DataProvider ABC + YFinanceProvider
├── strategy/
│   └── pairs.py            # Cointegration scan, z-score signals
├── portfolio/
│   └── construction.py     # Dollar-neutral weights, risk limits
├── backtest/
│   └── engine.py           # Walk-forward backtest loop
└── metrics/
    └── performance.py      # Sharpe, drawdown, turnover, beta
```

## Swapping Data Providers

The data layer uses an abstract `DataProvider` interface. To use Polygon, Alpaca, or any other source:

1. Create a new class in `data/provider.py` that subclasses `DataProvider`.
2. Implement `get_prices(tickers, start, end) -> pd.DataFrame`.
3. Register it in the `get_provider()` factory.
4. Set `PROVIDER` in `config.py`.

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | 50 S&P 500 names | Trading universe |
| `COINT_PVALUE` | 0.05 | Max p-value for cointegration |
| `ZSCORE_ENTRY` | 2.0 | Z-score threshold to enter a trade |
| `ZSCORE_EXIT` | 0.5 | Z-score threshold to exit |
| `TRAINING_WINDOW` | 252 | Training lookback (trading days) |
| `TRADING_WINDOW` | 63 | Out-of-sample trading period |
| `MAX_POSITION_WEIGHT` | 0.05 | Max weight per position leg |
| `MAX_GROSS_LEVERAGE` | 4.0 | Max sum of absolute weights |

## Performance Metrics

- Annualized return & volatility
- Sharpe ratio
- Maximum drawdown
- Annualized turnover
- Portfolio beta vs SPY
