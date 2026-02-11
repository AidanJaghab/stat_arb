"""
Performance metrics: Sharpe, drawdown, turnover, beta, and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def compute_metrics(
    backtest: pd.DataFrame,
    benchmark_prices: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute key performance stats from backtest results.

    Parameters
    ----------
    backtest : DataFrame with at least a 'portfolio_return' column.
    benchmark_prices : Series of benchmark adjusted closes (e.g. SPY)
                       for beta calculation.
    risk_free_rate : annualized risk-free rate (decimal).

    Returns a dict of metric_name -> value.
    """
    rets = backtest["portfolio_return"].dropna()
    n_days = len(rets)

    if n_days < 2:
        return {"error": "not enough data"}

    # --- Sharpe ratio (annualized) ---
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = rets - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0

    # --- Maximum drawdown ---
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # --- Annualized return ---
    total_return = cum.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1

    # --- Annualized volatility ---
    ann_vol = rets.std() * np.sqrt(252)

    # --- Turnover ---
    if "gross_leverage" in backtest.columns:
        # Approximate turnover as mean daily change in gross leverage
        turnover = backtest["gross_leverage"].diff().abs().mean() * 252
    else:
        turnover = np.nan

    # --- Beta vs benchmark ---
    beta = np.nan
    if benchmark_prices is not None:
        bench_ret = benchmark_prices.pct_change().dropna()
        common_idx = rets.index.intersection(bench_ret.index)
        if len(common_idx) > 10:
            r = rets.loc[common_idx]
            b = bench_ret.loc[common_idx]
            cov = np.cov(r, b)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "annualized_turnover": turnover,
        "beta": beta,
        "total_return": total_return,
        "n_trading_days": n_days,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics to the console."""
    print("\n" + "=" * 50)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    fmt = {
        "annualized_return": ("{:.2%}", "Ann. Return"),
        "annualized_volatility": ("{:.2%}", "Ann. Volatility"),
        "sharpe_ratio": ("{:.3f}", "Sharpe Ratio"),
        "max_drawdown": ("{:.2%}", "Max Drawdown"),
        "annualized_turnover": ("{:.2f}", "Ann. Turnover"),
        "beta": ("{:.4f}", "Beta (vs SPY)"),
        "total_return": ("{:.2%}", "Total Return"),
        "n_trading_days": ("{:d}", "Trading Days"),
    }
    for key, (f, label) in fmt.items():
        val = metrics.get(key, "N/A")
        if isinstance(val, float) and np.isnan(val):
            print(f"  {label:.<30s} N/A")
        elif isinstance(val, (int, float)):
            print(f"  {label:.<30s} {f.format(val)}")
        else:
            print(f"  {label:.<30s} {val}")
    print("=" * 50 + "\n")


def plot_equity_curve(backtest: pd.DataFrame, save_path: str = "equity_curve.png") -> None:
    """Plot and save the cumulative equity curve."""
    rets = backtest["portfolio_return"].dropna()
    cum = (1 + rets).cumprod()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    axes[0].plot(cum.index, cum.values, linewidth=1.2, color="#00ccff")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title("Stat-Arb Equity Curve")
    axes[0].grid(alpha=0.3)

    # Drawdown
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    axes[1].fill_between(dd.index, dd.values, 0, color="#ff4444", alpha=0.6)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Equity curve saved to {save_path}")
