# Filename: main.py

import pandas as pd
import numpy as np
from utils import winsorize_series, calculate_returns, market_impact_cost, borrow_cost
from backtest import walk_forward_backtest
from signal import generate_signals

# ========== 1. DATA PREPROCESSING ==========

def preprocess_data(price_df, currency='USD'):
    """
    price_df: DataFrame with MultiIndex (date, ticker)
    columns: ['adj_close', 'volume', 'bid', 'ask', 'currency']
    """
    df = price_df.copy()

    # Filter to reference currency
    df = df[df['currency'] == currency]

    # Handle missing values
    df['adj_close'] = df['adj_close'].fillna(method='ffill')
    df = df.dropna()

    # Winsorize adjusted prices and volumes
    df['adj_close'] = df.groupby('date')['adj_close'].transform(winsorize_series)
    df['volume'] = df.groupby('date')['volume'].transform(winsorize_series)

    # Mid price calculation
    df['mid_price'] = 0.5 * (df['bid'] + df['ask'])
    df['mid_price'] = df['mid_price'].fillna(df['adj_close'])

    return df


# ========== 2. RESEARCH RULES ==========

def apply_delisting_rule(prices, delist_info):
    for ticker, delist_date in delist_info.items():
        prices.loc[(prices.index.get_level_values('ticker') == ticker) & (prices.index.get_level_values('date') > delist_date), 'adj_close'] = np.nan
    return prices


# ========== 3. SIGNAL GENERATION ==========

def get_signals(df):
    signals = generate_signals(df)
    return signals


# ========== 4. BACKTESTING ==========

def run_backtest(price_df, signals):
    backtest_result = walk_forward_backtest(price_df, signals)
    return backtest_result


# ========== MAIN PIPELINE ==========

def main():
    # Load price data and delisting info
    price_df = pd.read_csv('data/price_data.csv', parse_dates=['date'])
    price_df.set_index(['date', 'ticker'], inplace=True)
    
    delist_info = {'XYZ': '2023-06-01'}  # Example delisting info

    # Preprocess data
    price_df = preprocess_data(price_df)
    price_df = apply_delisting_rule(price_df, delist_info)

    # Calculate daily returns
    price_df = calculate_returns(price_df)

    # Generate signals
    signals = get_signals(price_df)

    # Run backtest
    result = run_backtest(price_df, signals)

    # Display summary
    for k, v in result.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == '__main__':
    main()
