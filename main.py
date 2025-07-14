import numpy as np
import json
from tqdm import tqdm, trange
import os
import pandas as pd
import math
import pickle
import statistics
import matplotlib.pyplot as plt

# '''
# data
# '''

# #read
# df = pd.read_csv('../data.csv')

# #check unique tickers
# unique_tickers = df['ticker'].unique()
# print(f'num of unique stocks: {len(unique_tickers)}')

# #check unique dates
# unique_date = df['date'].unique()
# print(f'num of unique dates: {len(unique_date)}')

# #nan values
# nan_values = df.isna().any().any()
# if nan_values:
#     print("there are NaN values in the DataFrame.")
#     # df = df.fillna(method='ffill') # #fillin nan values
#     # df = df.drop(index=nan_rows.index[nan_rows].tolist()) #drop nan values
# else:
#     print("there are no NaN values in the DataFrame.")

# #sort data
# df = df.sort_values(['date', 'ticker'])



# '''
# signal functions
# '''
# def golden_death_cross_strategy(data, short_window=5, long_window=20):

#     # Calculate short-term and long-term moving averages
#     short_ma = data['last'].rolling(window=short_window, min_periods=1).mean()
#     long_ma = data['last'].rolling(window=long_window, min_periods=1).mean()

#     # Initialize the signals column with NaN values
#     data['golden_death_cross_signals'] = np.nan

#     # Set golden_death_cross_signals based on conditions, considering NaN values
#     data.loc[(short_ma.notna() & long_ma.notna()), 'golden_death_cross_signals'] = 0
#     data.loc[(short_ma > long_ma) & (short_ma.notna() & long_ma.notna()), 'golden_death_cross_signals'] = 1
#     data.loc[(short_ma < long_ma) & (short_ma.notna() & long_ma.notna()), 'golden_death_cross_signals'] = -1
    
#     return data


# def bollinger_bands_strategy(data, window=20, num_std_dev=2):
    
#     # Calculate the rolling mean and standard deviation
#     rolling_mean = data['last'].rolling(window=window).mean()
#     rolling_std = data['last'].rolling(window=window).std()

#     # Calculate the upper and lower Bollinger Bands
#     upper_band = rolling_mean + (rolling_std * num_std_dev)
#     lower_band = rolling_mean - (rolling_std * num_std_dev)

#     # Initialize the bollinger_bands_signals column with NaN values
#     data['bollinger_bands_signals'] = np.nan

#     # Set bollinger_bands_signals based on conditions, considering NaN values
#     data.loc[(data['last'].notna() & upper_band.notna() & lower_band.notna()), 'bollinger_bands_signals'] = 0  
#     data.loc[(data['last'] <= lower_band) & (data['last'].notna() & upper_band.notna() & lower_band.notna()), 'bollinger_bands_signals'] = 1
#     data.loc[(data['last'] >= upper_band) & (data['last'].notna() & upper_band.notna() & lower_band.notna()), 'bollinger_bands_signals'] = -1

#     return data


# def rsi_strategy(data, window=14, overbought=70, oversold=30):
#     # Calculate daily price changes
#     delta = data['last'].diff()

#     # Calculate gains (positive changes) and losses (negative changes)
#     gains = delta.where(delta > 0, 0)
#     losses = -delta.where(delta < 0, 0)

#     # Calculate the rolling average gains and losses
#     avg_gains = gains.rolling(window=window, min_periods=1).mean()
#     avg_losses = losses.rolling(window=window, min_periods=1).mean()

#     # Calculate the relative strength (RS) as the ratio of average gains to average losses
#     rs = avg_gains / avg_losses

#     # Calculate the RSI as 100 - (100 / (1 + RS))
#     rsi = 100 - (100 / (1 + rs))

#     # Initialize the rsi_signals column with NaN values
#     data['rsi_signals'] = np.nan

#     # Set rsi_signals based on conditions, considering NaN values
#     data.loc[(rsi.notna()), 'rsi_signals'] = 0  # Initialize with 0
#     data.loc[(rsi > overbought) & (rsi.notna()), 'rsi_signals'] = -1  # Overbought condition
#     data.loc[(rsi < oversold) & (rsi.notna()), 'rsi_signals'] = 1  # Oversold condition

#     return data


# def macd_strategy(data, short_window=12, long_window=26, signal_window=9):
#     # Calculate the short-term and long-term EMAs
#     short_ema = data['last'].ewm(span=short_window, adjust=False).mean()
#     long_ema = data['last'].ewm(span=long_window, adjust=False).mean()
    
#     # Calculate the MACD line
#     macd = short_ema - long_ema
    
#     # Calculate the signal line (EMA of MACD)
#     signal = macd.ewm(span=signal_window, adjust=False).mean()
    
#     # Initialize the signal column
#     data['macd_signal'] = 0
    
#     # Generate buy and sell signals
#     data['macd_signal'][short_window:] = np.where(macd[short_window:] > signal[short_window:], 1, 0)
#     data['macd_signal'][short_window:] = np.where(macd[short_window:] < signal[short_window:], -1, data['macd_signal'][short_window:])
    
#     return data


# def mean_reversion_strategy(data, window=10, threshold=1.5):
#     # Calculate the rolling mean using the specified window
#     rolling_mean = data['last'].rolling(window=window).mean()

#     # Initialize the signals list with zeros
#     signals = [0] * len(data)

#     # Set buy (1) and sell (-1) signals based on the threshold
#     for i in range(window, len(data)):
#         if data['last'].iloc[i] > rolling_mean.iloc[i] + threshold:
#             signals[i] = -1
#         elif data['last'].iloc[i] < rolling_mean.iloc[i] - threshold:
#             signals[i] = 1

#     # Add the 'mean_reversion_signals' column to the DataFrame
#     data['mean_reversion_signals'] = signals

#     return data


# def dual_moving_average_crossover_strategy(data, short_window=10, long_window=50):
#     # Initialize signals column (1 for buy, -1 for sell, 0 for no action)
#     data['dual_ma_signals'] = 0

#     for i in range(long_window, len(data)):
#         short_ma = data['last'].iloc[i - short_window:i].mean()
#         long_ma = data['last'].iloc[i - long_window:i].mean()

#         # Generate buy signals (short-term MA crosses above long-term MA)
#         if short_ma > long_ma and data['last'].iloc[i - 1] <= data['last'].iloc[i - long_window - 1]:
#             data.at[i, 'dual_ma_signals'] = 1

#         # Generate sell signals (short-term MA crosses below long-term MA)
#         elif short_ma < long_ma and data['last'].iloc[i - 1] >= data['last'].iloc[i - long_window - 1]:
#             data.at[i, 'dual_ma_signals'] = -1

#     return data


# volume_df = df.pivot(index='date', columns='ticker', values='volume')
# print(volume_df)

# df = golden_death_cross_strategy(df)
# df = bollinger_bands_strategy(df)
# df = rsi_strategy(df)
# df = macd_strategy(df)
# df = mean_reversion_strategy(df)
# df = dual_moving_average_crossover_strategy(df)
# df['final_signals'] = df['golden_death_cross_signals'] + df['bollinger_bands_signals'] + df['rsi_signals'] + df['macd_signal'] + df['mean_reversion_signals'] + df['dual_ma_signals']
# print(df)

# def signal_generator(df, signal_threshold=3, volume_quantile=0.7):
#     # Ensure correct types
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Step 1: Filter signals by absolute value
#     signal_extreme = df[np.abs(df['final_signals']) >= signal_threshold].copy()

#     # Step 2: Apply liquidity filter within each date
#     def filter_by_volume(group):
#         vol_threshold = group['volume'].quantile(volume_quantile)
#         return group[group['volume'] >= vol_threshold]

#     filtered = signal_extreme.groupby('date').apply(filter_by_volume).reset_index(drop=True)

#     return filtered[['ticker', 'date',  'last', 'volume', 'final_signals']]

# signal_df = signal_generator(df)
# print(signal_df)

signal_df = pd.read_csv('st1.csv')
signal_df = signal_df[['ticker', 'date', 'last', 'volume', 'final_signals']]



class Backtest:
    def __init__(self, dataframe, benchmark_code='s&p500', ini_cap=0):
        self.df = dataframe
        self.benchmark_code = benchmark_code
        self.short_cost_rate = 0.03
        self.initial_capital = ini_cap

    def run(self, start_date, end_date):
        print(f'begin to backtest {start_date} to {end_date}') 

        current_positions = {}
        trade_records = []
        daily_capitals = []
        daily_positions = []
        current_capital = self.initial_capital
        max_position = 0

        trading_dates = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]['date'].sort_values().unique()
        trading_dates = trading_dates.tolist()


        for date in trading_dates:
            date_str = date
            
            #get signals
            trade_df = self.df[self.df['date']==date]
            long_tickers = trade_df[trade_df['final_signals']>0]['ticker'].tolist()
            short_tickers = trade_df[trade_df['final_signals']<0]['ticker'].tolist()

            #record current positions
            current_position_count = len(current_positions)
            daily_positions.append(current_position_count)
            max_position = max(max_position, current_position_count)

            #execute signals
            for ticker in long_tickers: 
                if ticker not in current_positions:
                    ticker_close = trade_df[trade_df['ticker']==ticker]['last']
                    current_capital += -1*ticker_close

                    current_positions[ticker] = {
                        'position':'long',
                        'entry_price': ticker_close,
                        'entry_date': date_str
                    }

                    trade_record = {
                        "trade_date": date_str,
                        'stock_code': ticker,
                        'action': 'long',
                        'price': ticker_close,
                        "profit": 0,
                        'capatal_after_trade': current_capital
                    }
                    
                    trade_records.append(trade_record)
                
                elif ticker in current_positions and current_positions[ticker]['position'] == 'short':
                    ticker_close = trade_df[trade_df['ticker']==ticker]['last']
                    profit = current_positions[ticker]['entry_price'] - ticker_close
                    current_capital += 1*profit
                    
                    trade_record = {
                        "trade_date": date_str,
                        'stock_code': ticker,
                        'action': 'long',
                        'price': ticker_close,
                        "profit": profit,
                        'capatal_after_trade': current_capital
                    }
                    
                    trade_records.append(trade_record)
                    
                    del current_positions[ticker]

                    
            for ticker in short_tickers: 
                if ticker not in current_positions:
                    ticker_close = trade_df[trade_df['ticker']==ticker]['last']
                    current_capital += -self.short_cost_rate*ticker_close 

                    current_positions[ticker] = {
                        'position':'short',
                        'entry_price': (1 + self.short_cost_rate)*ticker_close ,
                        'entry_date': date_str
                    }

                    trade_record = {
                        "trade_date": date_str,
                        'stock_code': ticker,
                        'action': 'short',
                        'price': (1 + self.short_cost_rate)*ticker_close ,
                        "profit": 0,
                        'capatal_after_trade': current_capital
                    }
                    
                    trade_records.append(trade_record)
                
                elif ticker in current_positions and current_positions[ticker]['position'] == 'long':
                    ticker_close = trade_df[trade_df['ticker']==ticker]['last']
                    profit = ticker_close - current_positions[ticker]['entry_price']
                    current_capital += 1*profit
                    
                    trade_record = {
                        "trade_date": date_str,
                        'stock_code': ticker,
                        'action': 'short',
                        'price': ticker_close,
                        "profit": profit,
                        'capatal_after_trade': current_capital
                    }
                    
                    trade_records.append(trade_record)
                    
                    del current_positions[ticker]


            daily_capitals.append(current_capital)
            
            positions_value = sum(
                trade_df[trade_df['ticker']==key]['last'] for key in current_positions.keys() 
            )
            daily_positions.append(positions_value)

            print(f'{date} done')


        final_capital = daily_capitals[-1]
        

        #metrics

        #total_return
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        days = len(trading_dates)
        annual_return = (1 + total_return) ** (252/days) - 1

        print(daily_capitals)

        # #std
        # daily_returns = [(c-p)/p for c,p in zip(daily_capitals[1:], daily_capitals[:-1])]
        # vol = pd.Series(daily_returns).std() * (252 ** 0.5)

        # #sharpe
        # risk_free_rate = 0.03
        # sharpe_ratio = (annual_return - risk_free_rate) / vol if vol != 0 else 0

        # #max drawdown
        # peak = daily_capitals[0]
        # max_drawdown = 0 
        # for capital in daily_capitals:
        #     if capital > peak:
        #         peak = capital
        #     drawdown = (peak - capital) / peak
        #     max_drawdown = max(max_drawdown, drawdown)

        # print(annual_return, vol, sharpe_ratio, max_drawdown)

        # mean book size

        # hit rate

        # Daily turnover
        # Profit per doller traded


st1 = Backtest(dataframe=signal_df)  
st1.run('2013-01-04', '2013-12-31')       






# # Optional: filter for one ticker (if you have multiple)
# df['date'] = pd.to_datetime(df['date'])
# df_filtered = df[df['ticker'] == '2914 JT']
# plt.figure(figsize=(30, 5))
# plt.plot(df_filtered['date'], df_filtered['final_signals'], linestyle='-', color='blue')
# plt.title('Final Signals Over Time (2914 JT)')
# plt.xlabel('Date')
# plt.ylabel('Final Signal')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('test.jpg')