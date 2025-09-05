import yfinance as yf
import pandas as pd
from typing import List, Dict

def get_prices(tickers: List[str], start: str | None = None, end: str | None = None) -> Dict[str, pd.DataFrame]:
    """Download historical OHLCV data for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download (e.g., ['AAPL', 'MSFT']).
    start : str, optional
        Start date in 'YYYY-MM-DD' format.
    end : str, optional
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    dict
        A mapping from ticker to DataFrame of historical price data.
    """
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty:
            data[ticker] = df
    return data
