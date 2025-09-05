import pandas as pd
import requests
from typing import Optional

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_fred_series(series_id: str, start: str = "1900-01-01", end: Optional[str] = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch a macroeconomic time series from the FRED API.

    Parameters
    ----------
    series_id : str
        Identifier of the FRED series (e.g., 'CPIAUCSL').
    start : str, optional
        Start date in 'YYYY-MM-DD' format.
    end : str, optional
        End date in 'YYYY-MM-DD' format.
    api_key : str, optional
        FRED API key. Required for high volume.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'date' and 'value'.
    """
    params = {
        "series_id": series_id,
        "observation_start": start,
        "file_type": "json",
    }
    if end:
        params["observation_end"] = end
    if api_key:
        params["api_key"] = api_key

    response = requests.get(FRED_API_URL, params=params)
    response.raise_for_status()
    json_data = response.json()

    observations = json_data.get("observations", [])
    df = pd.DataFrame(observations)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]]
