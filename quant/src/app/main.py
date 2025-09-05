from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd

app = FastAPI(title="Quant API")


@app.get("/")
def read_root():
    return {"message": "Quant API is running"}


class WeightsRequest(BaseModel):
    predictions: Dict[str, float]
    top_n: int = 5
    short_n: int = 5
    cap_per_name: float = 0.05


@app.post("/portfolio")
def construct_portfolio_endpoint(req: WeightsRequest):
    """
    Construct a long/short portfolio from a dictionary of predictions.
    Returns a list of ticker-weight mappings.
    """
    df = pd.DataFrame(list(req.predictions.items()), columns=["Ticker", "Prediction"])
    # sort predictions descending
    df = df.sort_values(by="Prediction", ascending=False).reset_index(drop=True)
    longs = df.head(req.top_n).copy()
    shorts = df.tail(req.short_n).copy()
    longs["Weight"] = 1.0 / req.top_n
    shorts["Weight"] = -1.0 / req.short_n
    weights = pd.concat([longs[["Ticker", "Weight"]], shorts[["Ticker", "Weight"]]], ignore_index=True)
    # clip weights
    weights["Weight"] = weights["Weight"].clip(-req.cap_per_name, req.cap_per_name)
    # normalize long and short separately
    total_long = weights.loc[weights["Weight"] > 0, "Weight"].sum()
    total_short = -weights.loc[weights["Weight"] < 0, "Weight"].sum()
    if total_long > 0:
        weights.loc[weights["Weight"] > 0, "Weight"] /= total_long
    if total_short > 0:
        weights.loc[weights["Weight"] < 0, "Weight"] /= total_short
    return weights.to_dict(orient="records")
