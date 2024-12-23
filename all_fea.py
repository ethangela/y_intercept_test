

'''Feature Engineering Idea 2'''
def golden_death_cross_strategy(df, asset_id, columns, short_window=50, long_window=200):
    # Filter data for the specified asset_id
    data = df[df['asset_id'] == asset_id].copy()

    # Calculate short-term and long-term moving averages
    short_ma = data[columns].rolling(window=short_window, min_periods=1).mean()
    long_ma = data[columns].rolling(window=long_window, min_periods=1).mean()

    # Initialize a new DataFrame to store the signals
    signals_df = pd.DataFrame(index=data.index)

    # Set golden_death_cross_signals based on conditions, considering NaN values
    for col in columns:
        signals_df[f'{col}_gdc'] = 0
        signals_df.loc[short_ma[col] > long_ma[col], f'{col}_gdc'] = 1
        signals_df.loc[short_ma[col] < long_ma[col], f'{col}_gdc'] = -1

    return signals_df


def bollinger_bands_strategy(df, asset_id, columns, window=20, num_std_dev=2):
    # Filter data for the specified asset_id
    data = df[df['asset_id'] == asset_id].copy()
    
    # Calculate the rolling mean and standard deviation
    rolling_mean = data[columns].rolling(window=window).mean()
    rolling_std = data[columns].rolling(window=window).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    # Initialize a new DataFrame to store the signals
    signals_df = pd.DataFrame(index=data.index)

    # Set bollinger_bands_signals based on conditions, considering NaN values
    for col in columns:
        signals_df[f'{col}_bb'] = 0
        signals_df.loc[data[col] <= lower_band[col], f'{col}_bb'] = 1
        signals_df.loc[data[col] >= upper_band[col], f'{col}_bb'] = -1

    return signals_df


def rsi_strategy(df, asset_id, columns, window=14, overbought=70, oversold=30):

    data = df[ df['asset_id']==asset_id ].copy()

    # Calculate daily price changes
    delta = data[columns].diff()

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
    signals_df = pd.DataFrame(index=data.index)

    # Set rsi_signals based on conditions, considering NaN values
    for col in columns:
        signals_df[f'{col}_rsi'] = 0  # Initialize with 0
        signals_df.loc[rsi[col] > overbought, f'{col}_rsi'] = -1  # Overbought condition
        signals_df.loc[rsi[col] < oversold, f'{col}_rsi'] = 1  # Oversold condition
    return signals_df


def macd_strategy(df, asset_id, columns, short_window=12, long_window=26, signal_window=9): 

    data = df[ df['asset_id']==asset_id ].copy()

    # Calculate the short-term and long-term EMAs
    short_ema = data[columns].ewm(span=short_window, adjust=False).mean()
    long_ema = data[columns].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the MACD line
    macd = short_ema - long_ema
    
    # Calculate the signal line (EMA of MACD)
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    
    # Initialize the signal column
    signals_df = pd.DataFrame(index=data.index)
    
    # Generate buy and sell signals
    for col in columns: 
        signals_df[f'{col}_macd'] = 0  # Initialize with 0

        buy_signal = (macd[col] > signal[col]) & (macd[col].shift(1) <= signal[col].shift(1))
        sell_signal = (macd[col] < signal[col]) & (macd[col].shift(1) >= signal[col].shift(1))

        signals_df.loc[ buy_signal, f'{col}_macd' ] = 1
        signals_df.loc[ sell_signal, f'{col}_macd' ] = -1
    
    return signals_df


def mean_reversion_strategy(df, asset_id, columns, threshold=1.5):
    data = df[ df['asset_id']==asset_id ].copy()

    # Calculate the expanding mean and standard deviation
    expanding_mean = data[columns].expanding().mean()
    expanding_std = data[columns].expanding().std()

    # Calculate z-scores
    z_scores = (data[columns] - expanding_mean) / expanding_std

    # Initialize the signals column with zeros
    signals_df = pd.DataFrame(index=data.index)

    # Set buy (1) and sell (-1) signals based on the threshold
    for col in columns:
        signals_df[f'{col}_ms'] = 0  # Initialize with 0
        signals_df.loc[z_scores[col] > threshold, f'{col}_ms'] = -1  # Overvalued signal
        signals_df.loc[z_scores[col] < -threshold, f'{col}_ms'] = 1  # Undervalued signal


    return signals_df


def dual_moving_average_crossover_strategy(df, asset_id, columns, short_window=10, long_window=50):
    data = df[ df['asset_id']==asset_id ].copy()

    # Calculate short-term and long-term moving averages
    short_ma = data[columns].rolling(window=short_window).mean()
    long_ma = data[columns].rolling(window=long_window).mean()

    # Initialize signals column (1 for buy, -1 for sell, 0 for no action)
    signals_df = pd.DataFrame(index=data.index, columns=[f'{col}_dmac' for col in columns])

    for col in columns:
        signals_df[f'{col}_dmac'] = 0  # Initialize with 0

        # Generate buy signals (short-term MA crosses above long-term MA)
        buy_signals = (short_ma[col] > long_ma[col]) & (short_ma[col].shift(1) <= long_ma[col].shift(1))

        # Generate sell signals (short-term MA crosses below long-term MA)
        sell_signals = (short_ma[col] < long_ma[col]) & (short_ma[col].shift(1) >= long_ma[col].shift(1))

        # Assign buy and sell signals to the 'dual_ma_signals' column
        signals_df.loc[buy_signals, f'{col}_dmac'] = 1
        signals_df.loc[sell_signals, f'{col}_dmac'] = -1

    return signals_df


def william_strategy(df, asset_id, columns, window=14, overbought=-20, oversold=-80):
    # Filter data for the specified asset_id
    data = df[df['asset_id'] == asset_id].copy()

    # Calculate Highest_High and Lowest_Low    
    Highest_High = data[columns].rolling(window=window, min_periods=1).max()
    Lowest_Low = data[columns].rolling(window=window, min_periods=1).min()
    wiliams = -100 * (Highest_High - data[columns]) / (Highest_High - Lowest_Low)

    # Initialize a new DataFrame to store the signals
    signals_df = pd.DataFrame(index=data.index)

    # Set signals based on conditions
    for col in columns:
        signals_df[f'{col}_ws'] = 0
        signals_df.loc[wiliams[col] > overbought, f'{col}_ws'] = -1
        signals_df.loc[wiliams[col] < oversold, f'{col}_ws'] = 1

    return signals_df


def stochastic_oscillator_strategy(df, asset_id, columns, window=14, overbought=80, oversold=20):
    # Filter data for the specified asset_id
    data = df[df['asset_id'] == asset_id].copy()

    # Calculate the Stochastic Oscillator
    lowest_low = data[columns].rolling(window=window).min()
    highest_high = data[columns].rolling(window=window).max()
    k = 100 * ((data[columns] - lowest_low) / (highest_high - lowest_low))

    # Initialize a new DataFrame to store the signals
    signals_df = pd.DataFrame(index=data.index)

    # Set signals based on conditions
    for col in columns:
        signals_df[f'{col}_so'] = 0
        signals_df.loc[k[col] > overbought, f'{col}_so'] = -1
        signals_df.loc[k[col] < oversold, f'{col}_so'] = 1

    return signals_df



df1 = pd.read_parquet('./data/train-data-1.parquet')
df2 = pd.read_parquet('./data/train-data-2.parquet')
train = pd.concat([df1, df2], ignore_index=True)
train_length = len(train)
test = pd.read_parquet('./data/test-validation.parquet')
all_data = pd.concat([train, test], ignore_index=True) #TODO

features = all_data.columns[:185]
asset_ids = all_data['asset_id'].unique()

for i, ids in tqdm(enumerate(asset_ids), total=len(asset_ids)):
    df1 = golden_death_cross_strategy(all_data, ids, features)
    df2 = bollinger_bands_strategy(all_data, ids, features)
    df3 = rsi_strategy(all_data, ids, features)
    df4 = macd_strategy(all_data, ids, features)
    df5 = mean_reversion_strategy(all_data, ids, features)
    df6 = william_strategy(all_data, ids, features)
    df7 = stochastic_oscillator_strategy(all_data, ids, features)
    # # df8 = dual_moving_average_crossover_strategy(all_data, ids, features) #deleted

    save_dir = f'./data/save_{i}_{ids}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df1.to_parquet(os.path.join(save_dir, f'df1_{i}_{ids}.parquet'))
    df2.to_parquet(os.path.join(save_dir, f'df2_{i}_{ids}.parquet'))
    df3.to_parquet(os.path.join(save_dir, f'df3_{i}_{ids}.parquet'))
    df4.to_parquet(os.path.join(save_dir, f'df4_{i}_{ids}.parquet'))
    df5.to_parquet(os.path.join(save_dir, f'df5_{i}_{ids}.parquet'))
    df6.to_parquet(os.path.join(save_dir, f'df6_{i}_{ids}.parquet'))
    df7.to_parquet(os.path.join(save_dir, f'df7_{i}_{ids}.parquet'))

    if i == 0:
        all_data[df1.columns] = 0
        all_data[df2.columns] = 0 
        all_data[df3.columns] = 0 
        all_data[df4.columns] = 0 
        all_data[df5.columns] = 0 
        all_data[df6.columns] = 0 
        all_data[df7.columns] = 0 
    all_data.update(df1, overwrite=True)
    all_data.update(df2, overwrite=True)
    all_data.update(df3, overwrite=True)
    all_data.update(df4, overwrite=True)
    all_data.update(df5, overwrite=True)
    all_data.update(df6, overwrite=True)
    all_data.update(df7, overwrite=True)

all_data.to_parquet('./data/multi_all.parquet')
all_data.head(train_length).to_parquet('./data/multi_train.parquet')
all_data.tail(len(all_data)-train_length).reset_index(drop=True).to_parquet('./data/multi_test.parquet')

