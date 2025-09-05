import argparse
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor

from .data.prices import get_prices
from .features.technical import compute_returns, compute_rsi, compute_volatility


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate feature matrix and target vector from price DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price columns including 'Close'.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target returns over a 5-day horizon.
    """
    # Rename close column to uniform case if necessary
    if 'close' in df.columns:
        close = df['close']
    elif 'Close' in df.columns:
        close = df['Close']
    else:
        raise ValueError("Price DataFrame must contain a 'Close' column")

    features = pd.DataFrame(index=df.index)
    features['return_1d'] = compute_returns(close)
    features['rsi'] = compute_rsi(close)
    features['volatility'] = compute_volatility(features['return_1d'])
    target = close.pct_change(5).shift(-5)

    dataset = features.join(target.rename('target')).dropna()
    X = dataset.drop(columns='target')
    y = dataset['target']
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description='Train regression model for returns prediction.')
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to train on')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-out', type=str, default='model.json', help='Output path for saving the model')
    args = parser.parse_args()

    all_X = []
    all_y = []

    for ticker in args.tickers:
        # Fetch price data for each ticker separately
        df = get_prices([ticker], start=args.start, end=args.end)
        X, y = build_features(df)
        all_X.append(X)
        all_y.append(y)

    X_all = pd.concat(all_X)
    y_all = pd.concat(all_y)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
    model.fit(X_all, y_all)

    model.save_model(args.model_out)
    print(f'Model saved to {args.model_out}')


if __name__ == '__main__':
    main()
