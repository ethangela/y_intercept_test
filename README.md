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
- we can only long stocks initially, and we can only short stocks if we currently hold them. In simpler terms, we cannot engage in short selling, which involves borrowing stocks to short and repurchasing them later to cover the loan.
- we trade across all stocks. In other words, our portfolio contains all stocks provided.

Specifically, for each stock:
- Sum all trading signals generated from the strategies, and filter out signals with an absolute sum less than or equal to 2.
- Initialise `current_balance` = 0
- Initialise `current_position` = 0
- If `buy_signal > 2`:
  - If `current_position` is 0, buy `abs(buy_signal)` volumes with current price.
  - If `current_position` is greater than 0, buy additional volumes only when `abs(buy_signal)` is stronger.
- If `sell_signal > 2`:
  - If `current_position` is greater than 0, sell based on the strength of `abs(sell_signal)`.
  - If `current_position` is 0, do nothing (with potential modifications in the future, e.g. borrow stocks to short).

## Results (with running `output.py`)

With the trading strategies and executions described above, the project provides results showing the aggregate gains of the portfolio:
![Cumulative Gains Plot](cumulative_gains_plot.png)

## Future Work

Parameters in trading strategies, such as window size and threshold values, can be further fine-tuned given more time and extensive backtesting.
