##Filtros aplicados para decomposicao de serie temporal
##

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

import matplotlib.pyplot as plt
import seaborn as sns

# Set figure width to 12 and height to 9
plt.rcParams['figure.figsize'] = [12, 9]

df = pd.read_csv('data/brsr6_ponto.csv', index_col='DATA')
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df = df.resample('W').last()
series = df['FECHAMENTO']

cycle, trend = sm.tsa.filters.hpfilter(series, 50)
fig, ax = plt.subplots(3,1)
ax[0].plot(series)
ax[0].set_title('Price')
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[2].plot(cycle)
ax[2].set_title('Cycle')
plt.show()

result = STL(series).fit()
chart = result.plot()
plt.show()