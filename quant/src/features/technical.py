import pandas as pd
import numpy as np


def compute_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Compute percentage returns of a price series.

    Parameters
    ----------
    prices : pandas.Series
        Series of prices indexed by date.
    periods : int, optional
        Number of periods over which to compute returns. Default is 1 (simple daily returns).

    Returns
    -------
    pandas.Series
        Series of percentage returns.
    """
    return prices.pct_change(periods=periods)


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.

    Parameters
    ----------
    prices : pandas.Series
        Series of prices indexed by date.
    window : int, optional
        The lookback window for RSI calculation. Default is 14.

    Returns
    -------
    pandas.Series
        RSI values ranging from 0 to 100.
    """
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    gain = up.rolling(window=window).mean()
    loss = down.rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling volatility (standard deviation) of returns.

    Parameters
    ----------
    returns : pandas.Series
        Series of returns.
    window : int, optional
        The lookback window for volatility calculation. Default is 20.

    Returns
    -------
    pandas.Series
        Rolling standard deviation of the returns.
    """
    return returns.rolling(window=window).std()
