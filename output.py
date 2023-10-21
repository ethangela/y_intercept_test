import numpy as np
import json
from tqdm import tqdm, trange
import os
import pandas as pd
import math
import pickle
import statistics
import matplotlib.pyplot as plt

#read the new dataframe
df = pd.read_pickle('data_trade.pkl')
df.to_csv('data_trade.csv', index=False)

# Group the dataframe by 'date' and sum 'cumulative_gains' within each group
date_sum_gains = df.groupby('date')['cumulative_gains'].sum().reset_index()
filtered_df = date_sum_gains[date_sum_gains['date'] != 0]
filtered_df.to_csv('cumulative_gains.csv', index=False)

# Plot the cumulative gains against the date
plt.figure(figsize=(10, 8))
plt.plot(filtered_df.index, filtered_df['cumulative_gains'], marker='o', markersize=1, linestyle='-')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.7)  # Add a horizontal line at y=0
plt.title('Cumulative Gains Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Gains')
x_indices = np.arange(0, len(filtered_df), 300)
x_labels = filtered_df['date'].iloc[x_indices]
plt.xticks(x_indices, x_labels, rotation=20)
plt.savefig('cumulative_gains_plot.png')


