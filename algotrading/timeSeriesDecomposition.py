##Extrai o preço de fechamento de uma determinada serie temporal.
## É verificado nesta serie se ela é uma serie estacionaria, através de sua decomposição em partes que possuem tendências ou ciclos
##
import pandas as pd
import statsmodels.api as sm
# from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# include below line if you are using Jupyter Notebook
# %matplotlib inline

# Set figure width to 12 and height to 9
plt.rcParams['figure.figsize'] = [12, 9]

df = pd.read_csv('data/brsr6_ponto.csv', index_col='DATA')
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df = df.resample('W').last()
series = df['FECHAMENTO']

predictions = series.shift(1).dropna()
test_score = np.sqrt(mean_squared_error(series[int(len(series) * 0.7)+1:], predictions.iloc[int(len(series) * 0.7):]))
print('Test RMSE: %.5f' % test_score)

plt.plot(series.iloc[-25:], label='Price')
plt.plot(predictions[-25:], color='red', label='Prediction')
plt.legend()
plt.show()


fig, ax = plt.subplots()
ax = sns.regplot(series.iloc[-int(len(series) * 0.3):].pct_change(),
            predictions.iloc[-int(len(series) * 0.3):].pct_change(), )
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('EURUSD Observed vs Predicted Values')
ax.grid(True, which='both')
ax.axhline(y=0, color='#888888')
ax.axvline(x=0, color='#888888')
sns.despine(ax=ax, offset=0)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()

mae = round(abs(series.iloc[-int(len(series) * 0.3):].pct_change() - predictions.iloc[-int(len(series) * 0.3):].pct_change()).mean(),4)

print(f'The MAE is {mae}')


price_pred = pd.concat([series.iloc[-int(len(series) * 0.3):].pct_change(), predictions.iloc[-int(len(series) * 0.3):].pct_change()], axis=1)
price_pred.dropna(inplace=True)
price_pred.columns = ['Price', 'preds']

price_pred['hit'] = np.where(np.sign(price_pred['Price']) == np.sign(price_pred['preds']), 1, 0)

print(f"Hit rate: {round((price_pred['hit'].sum() / price_pred['hit'].count()) * 100,2)}%")