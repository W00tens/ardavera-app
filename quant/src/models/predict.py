import argparse
import pandas as pd
from xgboost import XGBRegressor

from .data.prices import get_prices
from .features.technical import compute_returns, compute_rsi, compute_volatility


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature DataFrame from price data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price columns.

    Returns
    -------
    pandas.DataFrame
        Feature DataFrame with return, RSI and volatility columns.
    """
    if 'close' in df.columns:
        close = df['close']
    elif 'Close' in df.columns:
        close = df['Close']
    else:
        raise ValueError("Price DataFrame must contain a 'Close' column")

    feats = pd.DataFrame(index=df.index)
    feats['return_1d'] = compute_returns(close)
    feats['rsi'] = compute_rsi(close)
    feats['volatility'] = compute_volatility(feats['return_1d'])
    return feats.dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate predictions using a trained model.')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to predict for')
    parser.add_argument('--asof', type=str, required=True, help='Date up to which to fetch data (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='model.json', help='Path to trained model file')
    args = parser.parse_args()

    model = XGBRegressor()
    model.load_model(args.model)

    for ticker in args.tickers:
        df = get_prices([ticker], start=None, end=args.asof)
        X = build_features(df)
        if X.empty:
            print(f'{ticker}: no features available for prediction')
            continue
        latest = X.iloc[[-1]]
        pred = model.predict(latest)[0]
        print(f'{ticker}: {pred}')


if __name__ == '__main__':
    main()
