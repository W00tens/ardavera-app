import requests
import pandas as pd
from typing import Optional


def fetch_company_facts(cik: str, concept: str, units: Optional[str] = None, user_agent: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch reported financial facts for a given company from the SEC XBRL API.

    Parameters
    ----------
    cik : str
        The Central Index Key identifying the company. It will be zero-padded to ten digits.
    concept : str
        The financial concept to retrieve (e.g., 'Revenues', 'Assets').
    units : str, optional
        Measurement units (e.g., 'USD', 'shares'). If None, the first available unit will be used.
    user_agent : str, optional
        A descriptive User-Agent string as required by the SEC API. If not provided, a generic one is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing fact data with columns such as 'end' and 'val'.
    """
    # Zero-pad CIK to 10 digits
    cik_str = f"{int(cik):010d}"
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"

    headers = {"User-Agent": user_agent or "quant-prototype-app"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json().get("facts", {})
    if concept not in data:
        return pd.DataFrame()

    concept_data = data[concept].get("units", {})
    # Select the specified units or default to the first available unit key
    unit_key = units or (next(iter(concept_data.keys())) if concept_data else None)
    fact_list = concept_data.get(unit_key, []) if unit_key else []

    df = pd.DataFrame(fact_list)
    # Convert date columns
    for date_col in ["end", "start"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
    return df
