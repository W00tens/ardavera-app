import pandas as pd
import argparse
from ..data.prices import get_prices


def backtest_portfolio(weights_csv: str, start: str, end: str, output_path: str = None) -> pd.DataFrame:
    """
    Backtest a static long/short portfolio over a time range.
    
    :param weights_csv: Path to CSV with 'Ticker' and 'Weight' columns
    :param start: Start date (YYYY-MM-DD)
    :param end: End date (YYYY-MM-DD)
    :param output_path: Optional path to save equity curve as CSV
    :return: DataFrame with Date, Returns, and Equity columns
    """
    weights = pd.read_csv(weights_csv)
    tickers = weights['Ticker'].tolist()
    prices = get_prices(tickers, start, end)
    returns = prices.pct_change().dropna()
    # Map weights to series
    weight_series = pd.Series(weights['Weight'].values, index=weights['Ticker'])
    # Align series to returns columns
    portfolio_returns = returns.mul(weight_series, axis=1).sum(axis=1)
    equity = (1 + portfolio_returns).cumprod()
    result = pd.DataFrame({'Date': portfolio_returns.index, 'Returns': portfolio_returns.values, 'Equity': equity.values})
    if output_path:
        result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Backtest a static long/short portfolio.")
    parser.add_argument('--weights', type=str, required=True, help='Path to CSV containing weights')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None, help='Path to save equity curve CSV')
    args = parser.parse_args()
    result = backtest_portfolio(args.weights, args.start, args.end, args.output)
    if args.output is None:
        print(result)


if __name__ == '__main__':
    main()
