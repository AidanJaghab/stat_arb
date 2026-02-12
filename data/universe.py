"""
Fetch the top ~1000 US equity tickers (Russell 1000 proxy).
Uses S&P 500 + S&P 400 MidCap from Wikipedia, plus additional large-caps.
"""

import pandas as pd


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def get_sp400_tickers() -> list[str]:
    """Scrape current S&P 400 MidCap constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    tables = pd.read_html(url)
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else "Ticker symbol"
    tickers = df[col].str.replace(".", "-", regex=False).tolist()
    return tickers


# Additional large/mid-cap names not always in S&P indices
_EXTRA_TICKERS = [
    "PLTR", "COIN", "HOOD", "SOFI", "RBLX", "U", "DKNG", "LCID",
    "RIVN", "DNA", "IONQ", "JOBY", "GRAB", "SE", "SHOP", "SNAP",
    "PINS", "ROKU", "Z", "ZG", "ETSY", "W", "CHWY", "DUOL",
    "RDDT", "ARM", "BIRK", "CART", "CAVA", "TOST", "DLO", "NU",
    "XPEV", "NIO", "LI", "BABA", "JD", "PDD", "BIDU", "TME",
    "BILI", "IQ", "TAL", "EDU", "FUTU", "TIGR", "WB", "ZH",
    "OPEN", "CVNA", "UPST", "AFRM", "LMND", "ROOT", "OSCR", "HIMS",
    "DOCS", "TDOC", "AMWL", "ACCD", "TALK", "GDRX", "RXRX", "DNAY",
    "SMCI", "VRT", "CEG", "VST", "OKLO", "NNE", "SMR", "LEU",
    "CCJ", "UEC", "DNN", "NXE", "UUUU", "RIG", "VAL", "HP",
    "PSTG", "NET", "CRWD", "ZS", "FTNT", "S", "CYBR", "TENB",
    "RPD", "VRNS", "QLYS", "SAIC", "BAH", "LDOS", "CACI", "KBR",
]


def get_top_universe(target: int = 1000) -> list[str]:
    """
    Return a deduplicated list of ~*target* US equity tickers,
    combining S&P 500 + S&P 400 + extra names.
    """
    tickers: list[str] = []

    try:
        tickers.extend(get_sp500_tickers())
        print(f"  Loaded {len(tickers)} S&P 500 tickers")
    except Exception as e:
        print(f"  Warning: could not fetch S&P 500 list: {e}")

    try:
        mid = get_sp400_tickers()
        tickers.extend(mid)
        print(f"  Loaded {len(mid)} S&P 400 MidCap tickers")
    except Exception as e:
        print(f"  Warning: could not fetch S&P 400 list: {e}")

    tickers.extend(_EXTRA_TICKERS)

    # Deduplicate, preserve order
    seen = set()
    unique = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"  Total unique tickers: {len(unique)}")
    if len(unique) > target:
        unique = unique[:target]
    return unique
