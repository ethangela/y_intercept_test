# y_intercept_test

## Workflow (with running `trade.py`)

This project follows a structured workflow for generating statistical trading signals and implementing trading strategies. The key steps in the workflow are as follows:

### 1. Preprocessing Data

Check for missing values (NA values) and negative values.

### 2. Generating Statistical Trading Signals

The project uses various statistical trading signal strategies to make trading decisions, where each strategy generates buy and sell signals based on specific conditions:

#### a. Moving Average Crossover Strategy

- Buy when a short-term moving average (5-day) crosses above a long-term moving average (20-day).
- Sell when the short-term moving average crosses below the long-term moving average.

#### b. Bollinger Bands Strategy

- Buy when the price touches or falls below the lower Bollinger Band (20-day, 2-std) and then moves back into the band.
- Sell when the price touches or rises above the upper Bollinger Band and then moves back into the band.

#### c. Relative Strength Index (RSI) Strategy

- Buy when the RSI crosses above an oversold threshold (30), indicating a potential upward move.
- Sell when the RSI crosses below an overbought threshold (70), indicating a potential downward move.

#### d. Moving Average Convergence Divergence (MACD) Strategy

- Buy when the MACD line crosses above the signal line.
- Sell when the MACD line crosses below the signal line.

#### e. Mean Reversion Strategy

- Buy when the stock's price is significantly below its moving average, indicating it might revert to the mean.
- Sell when the stock's price is significantly above its moving average.

#### f. Dual Moving Average Crossover Strategy

- Use two moving averages (10-day and 50-day). Buy when the short-term MA crosses above the long-term MA and sell when it crosses below.

### 3. Trading Strategies

After generating trading signals, we combine (sum) all the trading signals from the different strategies and execute a trading plan with two simple setups for this tiny project: 
- In the beginning, we can long stocks and borrow stocks to short, for which we repurchase them later to cover the loan.
- we trade across all stocks. In other words, our portfolio contains all the stocks provided.

Specifically, for each stock:
- Sum all trading signals generated from the strategies and filter out the aggregated `buy_signal` (`sell_signal`) whose absolute value is less than or equal to 2 (1).

- Initialize `current_balance`, whcih represents the current balance available for trading, to 0.
- Initialize `current_position`, which indicates the current stock position held in the portfolio, to 0.

- If a `buy_signal` is presented:
  - If `current_position` is 0, we long the stock with the current stock price and unit volume, reducing `current_balance` and resulting in positive `current_position`.
  - If `current_position` is negative, we short all currently-hold stocks with the current stock price, increasing `current_balance` and resetting `current_position` to 0.
  - If `current_position` is positive, we do nothing.

- If a `sell_signal` is presented:
  - If `current_position` is 0, we short the stock with the current stock price and unit volume, increasing `current_balance` and resulting in negative `current_position`.
  - If `current_position` is positive, we long all currently-hold stocks with the current stock price, reducing `current_balance` and resetting `current_position` to 0.
  - If `current_position` is negative, we do nothing.
 
- We do not stop the procedure above until `current_balance` is greater than a pre-defined threshold, where a larger threshold indicates a more aggressive trading taste. 
 

## Future Work

Parameters in trading strategies, such as window size and threshold values, and those in trading executions, such as trading volume and stop-rule threshold, can be further fine-tuned given more time and extensive backtesting.


## Results (with running `output.py`)

With the trading strategies and executions described above, the project provides results showing the aggregate gains of the portfolio:

stop-rule threshold 200:
![Cumulative Gains Plot](cumulative_gains_200.png)

stop-rule threshold 1000:
![Cumulative Gains Plot](cumulative_gains_1000.png)
