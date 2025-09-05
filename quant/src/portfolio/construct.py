import pandas as pd
import numpy as np
import argparse


def construct_portfolio(predictions_csv: str, top_n: int = 5, short_n: int = 5, cap_per_name: float = 0.05, output_path: str = None) -> pd.DataFrame:
    """
    Construct a long/short portfolio based on predictions.
    
    :param predictions_csv: Path to CSV with columns 'Ticker' and 'Prediction'
    :param top_n: Number of top predicted tickers to go long
    :param short_n: Number of bottom predicted tickers to short
    :param cap_per_name: Maximum absolute weight per ticker
    :param output_path: Optional path to save resulting weights to CSV
    :return: DataFrame with 'Ticker' and 'Weight'
    """
    predictions = pd.read_csv(predictions_csv)
    # sort by prediction descending
    predictions = predictions.sort_values(by='Prediction', ascending=False).reset_index(drop=True)
    longs = predictions.head(top_n).copy()
    shorts = predictions.tail(short_n).copy()
    # initial equal weights
    longs['Weight'] = 1.0 / top_n
    shorts['Weight'] = -1.0 / short_n
    weights = pd.concat([longs[['Ticker', 'Weight']], shorts[['Ticker', 'Weight']]], ignore_index=True)
    # clip weights to cap_per_name
    weights['Weight'] = weights['Weight'].clip(-cap_per_name, cap_per_name)
    # normalize long and short sides separately
    total_long = weights.loc[weights['Weight'] > 0, 'Weight'].sum()
    total_short = -weights.loc[weights['Weight'] < 0, 'Weight'].sum()
    if total_long > 0:
        weights.loc[weights['Weight'] > 0, 'Weight'] /= total_long
    if total_short > 0:
        weights.loc[weights['Weight'] < 0, 'Weight'] /= total_short
    # Save if output path provided
    if output_path:
        weights.to_csv(output_path, index=False)
    return weights


def main():
    parser = argparse.ArgumentParser(description="Construct a long/short portfolio from prediction CSV.")
    parser.add_argument('--predictions', type=str, required=True, help='Path to CSV containing predictions')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top tickers to go long')
    parser.add_argument('--short_n', type=int, default=5, help='Number of bottom tickers to short')
    parser.add_argument('--cap_per_name', type=float, default=0.05, help='Maximum absolute weight per name')
    parser.add_argument('--output', type=str, default=None, help='Path to save the resulting weights CSV')
    args = parser.parse_args()
    weights = construct_portfolio(args.predictions, args.top_n, args.short_n, args.cap_per_name, args.output)
    if args.output is None:
        print(weights)


if __name__ == '__main__':
    main()
