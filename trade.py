# -*- coding: utf-8 -*-
import numpy as np
import json
from tqdm import tqdm, trange
import os
import pandas as pd
import math
import pickle
import statistics


'''
data pre-processing
'''
#read data
df = pd.read_csv('data.csv')
# print(df.info)

# check unique stocks
unique_tickers = df['ticker'].unique()
print(f'num of unique stocks: {len(unique_tickers)}')

#check nan values
nan_values = df.isna().any().any()
if nan_values:
    print("there are NaN values in the DataFrame.")
    #actions below can be uncommented if NaN values exist
    # df = df.fillna(method='ffill') # #fillin nan values
    # df = df.drop(index=nan_rows.index[nan_rows].tolist()) #drop nan values
else:
    print("there are no NaN values in the DataFrame.")

#check negative values
for col in ['last','volume']:
    ng_id_list = df[df[col]<0].index.values
    if ng_id_list:
        print("there are negative price/volume values in the DataFrame.")
        #actions below can be uncommented if negative values exist
        # df.loc[ng_id_list,col] = -1 * df.loc[ng_id_list,col]
    else:
        print("there are no negative values in the DataFrame.")




'''
trading
'''
#define strategies functions
def golden_death_cross_strategy(data, short_window, long_window):

    # Calculate short-term and long-term moving averages
    short_ma = data['last'].rolling(window=short_window, min_periods=1).mean()
    long_ma = data['last'].rolling(window=long_window, min_periods=1).mean()

    # Initialize the signals column with NaN values
    data['golden_death_cross_signals'] = np.nan

    # Set golden_death_cross_signals based on conditions, considering NaN values
    data.loc[(short_ma.notna() & long_ma.notna()), 'golden_death_cross_signals'] = 0
    data.loc[(short_ma > long_ma) & (short_ma.notna() & long_ma.notna()), 'golden_death_cross_signals'] = 1
    data.loc[(short_ma < long_ma) & (short_ma.notna() & long_ma.notna()), 'golden_death_cross_signals'] = -1
    
    return data


def bollinger_bands_strategy(data, window, num_std_dev):
    
    # Calculate the rolling mean and standard deviation
    rolling_mean = data['last'].rolling(window=window).mean()
    rolling_std = data['last'].rolling(window=window).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    # Initialize the bollinger_bands_signals column with NaN values
    data['bollinger_bands_signals'] = np.nan

    # Set bollinger_bands_signals based on conditions, considering NaN values
    data.loc[(data['last'].notna() & upper_band.notna() & lower_band.notna()), 'bollinger_bands_signals'] = 0  
    data.loc[(data['last'] <= lower_band) & (data['last'].notna() & upper_band.notna() & lower_band.notna()), 'bollinger_bands_signals'] = 1
    data.loc[(data['last'] >= upper_band) & (data['last'].notna() & upper_band.notna() & lower_band.notna()), 'bollinger_bands_signals'] = -1

    return data


def rsi_strategy(data, window, overbought=70, oversold=30):
    # Calculate daily price changes
    delta = data['last'].diff()

    # Calculate gains (positive changes) and losses (negative changes)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate the rolling average gains and losses
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()

    # Calculate the relative strength (RS) as the ratio of average gains to average losses
    rs = avg_gains / avg_losses

    # Calculate the RSI as 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + rs))

    # Initialize the rsi_signals column with NaN values
    data['rsi_signals'] = np.nan

    # Set rsi_signals based on conditions, considering NaN values
    data.loc[(rsi.notna()), 'rsi_signals'] = 0  # Initialize with 0
    data.loc[(rsi > overbought) & (rsi.notna()), 'rsi_signals'] = -1  # Overbought condition
    data.loc[(rsi < oversold) & (rsi.notna()), 'rsi_signals'] = 1  # Oversold condition

    return data


def macd_strategy(data, short_window, long_window, signal_window):
    # Calculate the short-term and long-term EMAs
    short_ema = data['last'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['last'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the MACD line
    macd = short_ema - long_ema
    
    # Calculate the signal line (EMA of MACD)
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    
    # Initialize the signal column
    data['macd_signal'] = 0
    
    # Generate buy and sell signals
    data['macd_signal'][short_window:] = np.where(macd[short_window:] > signal[short_window:], 1, 0)
    data['macd_signal'][short_window:] = np.where(macd[short_window:] < signal[short_window:], -1, data['macd_signal'][short_window:])
    
    return data


def mean_reversion_strategy(data, window, threshold):
    # Calculate the rolling mean using the specified window
    rolling_mean = data['last'].rolling(window=window).mean()

    # Initialize the signals list with zeros
    signals = [0] * len(data)

    # Set buy (1) and sell (-1) signals based on the threshold
    for i in range(window, len(data)):
        if data['last'].iloc[i] > rolling_mean.iloc[i] + threshold:
            signals[i] = -1
        elif data['last'].iloc[i] < rolling_mean.iloc[i] - threshold:
            signals[i] = 1

    # Add the 'mean_reversion_signals' column to the DataFrame
    data['mean_reversion_signals'] = signals

    return data


def dual_moving_average_crossover_strategy(data, short_window, long_window):
    # Initialize signals column (1 for buy, -1 for sell, 0 for no action)
    data['dual_ma_signals'] = 0

    for i in range(long_window, len(data)):
        short_ma = data['last'].iloc[i - short_window:i].mean()
        long_ma = data['last'].iloc[i - long_window:i].mean()

        # Generate buy signals (short-term MA crosses above long-term MA)
        if short_ma > long_ma and data['last'].iloc[i - 1] <= data['last'].iloc[i - long_window - 1]:
            data.at[i, 'dual_ma_signals'] = 1

        # Generate sell signals (short-term MA crosses below long-term MA)
        elif short_ma < long_ma and data['last'].iloc[i - 1] >= data['last'].iloc[i - long_window - 1]:
            data.at[i, 'dual_ma_signals'] = -1

    return data


#apply the strategies on each stock
count = 0
for stock in tqdm(unique_tickers.tolist()):

    data = df[df['ticker']==stock]

    #define strategy parameters #TODO: can be fine-tuned in the future
    golden_death_short_window = 5
    golden_death_long_window = 20
    
    bollinger_window = 20
    num_std_dev = 2
    
    rsi_window = 14
    oversold_threshold = 30
    overbought_threshold = 70

    macd_short_window = 12
    macd_long_window = 26
    macd_signal_window = 9

    reversion_window = 10
    reversion_threshold = 1.5

    dual_short_window = 10
    dual_long_window = 50

    #apply each strategy
    data = golden_death_cross_strategy(data, golden_death_short_window, golden_death_long_window)
    data = bollinger_bands_strategy(data, bollinger_window, num_std_dev)
    data = rsi_strategy(data, rsi_window, oversold_threshold, overbought_threshold)
    data = macd_strategy(data, macd_short_window, macd_long_window, macd_signal_window)
    data = mean_reversion_strategy(data, reversion_window, reversion_threshold)
    data = dual_moving_average_crossover_strategy(data, dual_short_window, dual_long_window)

    #fill na values with 0 (0 indicates no action)
    data = data.fillna(0)

    #combine signals from all strategies into a single 'combined_signals' column
    data['combined_signal'] = (
        data['golden_death_cross_signals'] +
        data['bollinger_bands_signals'] +
        data['rsi_signals'] +
        data['macd_signal'] +
        data['mean_reversion_signals'] +
        data['dual_ma_signals']
    )

    #filter out small trading signals #TODO: threshold 2 can be fine-tuned in the future
    data['trade_signal'] = data['combined_signal'].apply(lambda x: x if x > 2 or x < -1 else 0)
    
    #trade actions
    def calculate_cumulative_gains(df):
        cumulative_gains = []
        cumulative_stocks = []
        current_balance = 0
        current_position = 0  #keeps track of the current position (positive for buy, negative for sell)
        price_at_entry = 0  #price at which we entered a position

        for _, row in df.iterrows():
            
            if current_balance > 1000: #TODO tune-able parameter, where a larger value indicates a more aggressive strategy 
                current_position = current_position
                current_balance = current_balance
            
            else:
                if row['trade_signal'] > 0:  #buy signal
                    if current_position == 0: 
                        current_position += row['trade_signal']
                        price_at_entry = row['last']
                        current_balance -= current_position * price_at_entry
                    elif current_position > 0: 
                        current_position = current_position
                        current_balance = current_balance
                    elif current_position < 0: 
                        price_at_entry = row['last']
                        current_balance -= current_position * price_at_entry
                        current_position = 0

                elif row['trade_signal'] < 0:  #sell signal
                    if current_position == 0: 
                        current_position += row['trade_signal']
                        price_at_entry = row['last']
                        current_balance += current_position * price_at_entry  
                    elif current_position > 0: 
                        price_at_entry = row['last']
                        current_balance += current_position * price_at_entry
                        current_position = 0
                    elif current_position < 0: 
                        current_position = current_position
                        current_balance = current_balance

            cumulative_gains.append(current_balance)
            cumulative_stocks.append(current_position)

        df['cumulative_gains'] = cumulative_gains
        df['cumulative_stocks'] = cumulative_stocks

    calculate_cumulative_gains(data)

    #save the results
    if count == 0:
        data.to_pickle('data_new.pkl')
    else:
        df_final = pd.read_pickle('data_new.pkl')
        df_final = pd.concat([df_final, data], ignore_index=True)
        df_final.to_pickle('data_new.pkl')
    count += 1


    


# ##output
# print('features expanded!. Samples shown below:')
# print(df.head(10))
# df.to_pickle('data_trade_signal.pkl')
# print('file saved')

