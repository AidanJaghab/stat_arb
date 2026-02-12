"""
Fetch GICS sector classifications for S&P 500 constituents.
"""

import io
import requests
import pandas as pd

_HEADERS = {"User-Agent": "stat-arb-bot/1.0 (educational project)"}


def get_sp500_sectors() -> dict[str, str]:
    """
    Return a dict mapping ticker -> GICS sector for S&P 500 companies.
    Scraped from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return dict(zip(df["Symbol"], df["GICS Sector"]))


# Fallback sector map for the most liquid names (used if Wikipedia fails)
SECTOR_MAP_FALLBACK = {
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    "OXY": "Energy", "HAL": "Energy", "DVN": "Energy", "HES": "Energy",
    # Tech
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "AMD": "Information Technology", "INTC": "Information Technology",
    "CRM": "Information Technology", "ADBE": "Information Technology",
    "CSCO": "Information Technology", "TXN": "Information Technology",
    "QCOM": "Information Technology", "AMAT": "Information Technology",
    "MU": "Information Technology", "LRCX": "Information Technology",
    "KLAC": "Information Technology", "MCHP": "Information Technology",
    "ADI": "Information Technology", "SNPS": "Information Technology",
    "CDNS": "Information Technology", "NXPI": "Information Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "USB": "Financials",
    "PNC": "Financials", "TFC": "Financials", "AXP": "Financials",
    "CME": "Financials", "ICE": "Financials", "MCO": "Financials",
    "SPGI": "Financials", "MMC": "Financials", "CB": "Financials",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    "KDP": "Consumer Staples", "STZ": "Consumer Staples", "KHC": "Consumer Staples",
    "GIS": "Consumer Staples", "HSY": "Consumer Staples", "SJM": "Consumer Staples",
    # Healthcare
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "PFE": "Health Care", "MRK": "Health Care", "ABBV": "Health Care",
    "TMO": "Health Care", "ABT": "Health Care", "DHR": "Health Care",
    "BMY": "Health Care", "AMGN": "Health Care", "GILD": "Health Care",
    "ISRG": "Health Care", "MDT": "Health Care", "SYK": "Health Care",
    "ZTS": "Health Care", "BSX": "Health Care", "VRTX": "Health Care",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services",
    "DIS": "Communication Services", "NFLX": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "MAR": "Consumer Discretionary",
    # Industrials
    "HON": "Industrials", "UPS": "Industrials", "CAT": "Industrials",
    "DE": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "GE": "Industrials", "BA": "Industrials", "MMM": "Industrials",
    "FDX": "Industrials", "WM": "Industrials", "EMR": "Industrials",
    "ITW": "Industrials", "ETN": "Industrials", "NSC": "Industrials",
    "CSX": "Industrials", "UNP": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    "SRE": "Utilities", "XEL": "Utilities", "WEC": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "SPG": "Real Estate",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "FCX": "Materials", "NEM": "Materials",
    "NUE": "Materials", "DOW": "Materials",
}


def get_sectors() -> dict[str, str]:
    """Return ticker -> sector mapping, trying Wikipedia first."""
    try:
        return get_sp500_sectors()
    except Exception:
        return SECTOR_MAP_FALLBACK
