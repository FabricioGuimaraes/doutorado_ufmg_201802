##Teste de uma estrageia para trade
##

import pandas as pd
import numpy as np
from pandas_datareader import data
from math import sqrt
import matplotlib.pyplot as plt

# download data into DataFrame and create moving averages columns
sp500 = data.DataReader('^GSPC', 'yahoo', start='1/1/2014')
sp500['42d'] = np.round(sp500['Close'].rolling(window=42).mean(), 2)
sp500['252d'] = np.round(sp500['Close'].rolling(window=252).mean(), 2)

# create column with moving average spread differential
sp500['42-252'] = sp500['42d'] - sp500['252d']

# set desired number of points as threshold for spread difference and create column containing strategy 'Stance'
X = 50
days = 50
sp500['Stance'] = np.where(sp500['42-252'] > X, 1, 0)
sp500['Stance'] = np.where(sp500['42-252'] < -X, -1, sp500['Stance'])
sp500['Stance'].value_counts()
# create columns containing daily market log returns and strategy daily log returns
sp500['Market Returns'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500['Strategy'] = sp500['Market Returns'] * sp500['Stance'].shift(1)
# set up a new column to hold our stance relevant for the prespecified holding period
sp500['Stance2'] = 0
# set out predetermined holding period, after which time we will go back to holding cash and wait
# for the next moving average cross over - also we will ignore any extra crossovers during this holding period days = 50
# iterate through the DataFrame and update the "Stance2" column to hold the revelant stance
for i in range(X, len(sp500)):
    # logical test to check for 1) a cross over short over long MA 2) That we are currently in cash
    if (sp500['Stance'].iloc[i] > sp500['Stance'].iloc[i - 1]) and (sp500['Stance'].iloc[i - 1] == 0) and (
            sp500['Stance2'].iloc[i - 1] == 0):
        # populate the DataFrame forward in time for the amount of days in our holding period
        for k in range(days):
            try:
                sp500['Stance2'].iloc[i + k] = 1
                sp500['Stance2'].iloc[i + k + 1] = 0
            except:
                pass
    # logical test to check for 1) a cross over short under long MA 2) That we are currently in cash
    if (sp500['Stance'].iloc[i] < sp500['Stance'].iloc[i - 1]) and (sp500['Stance'].iloc[i - 1] == 0) and (
            sp500['Stance2'].iloc[i - 1] == 0):
        # populate the DataFrame forward in time for the amount of days in our holding period
        for k in range(days):
            try:
                sp500['Stance2'].iloc[i + k] = -1
                sp500['Stance2'].iloc[i + k + 1] = 0
            except:
                pass

# Calculate daily market returns and strategy daily returns
sp500['Market Returns'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500['Strategy'] = sp500['Market Returns'] * sp500['Stance2'].shift(1)

# plot strategy returns vs market returns
sp500[['Market Returns', 'Strategy']].cumsum().plot(grid=True, figsize=(8, 5))
plt.show()

# set strategy starting equity to 1 (i.e. 100%) and generate equity curve
sp500['Strategy Equity'] = sp500['Strategy'].cumsum() + 1

# show chart of equity curve
sp500['Strategy Equity'].plot(grid=True, figsize=(8, 5))
plt.show()