##Uso de alguns dos principais indicadores tecnicos para trade
##

import pandas_datareader.data as web
import pandas as pd
import numpy as np
from talib import BBANDS, EMA, WMA, MACD, RSI, MFI, AD, ADOSC, OBV
import matplotlib.pyplot as plt

##Overlap Studies
brsr6CSV = pd.read_csv("data/brsr6_ponto.csv")
abertura = brsr6CSV['ABERTURA'].values
fechamento = brsr6CSV['FECHAMENTO'].values
abertura = brsr6CSV['ABERTURA'].values
minimo = brsr6CSV['MINIMO'].values
maximo = brsr6CSV['MAXIMO'].values
variacao = brsr6CSV['VARIACAO'].values
volume = brsr6CSV['VOLUME'].values

upBR, midBR, lowBR = BBANDS(fechamento, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
ema = EMA(fechamento, timeperiod=30) #curto prazo 21. Medio prazo 24. Longo prazo 89
wma = WMA(fechamento, timeperiod=30)


# Momentum Indicators
macd, macdsignal, macdhist = MACD(fechamento, fastperiod=12, slowperiod=26, signalperiod=9)
# print('-----------------------------------MACD-----------------------------------')
# print(macd[34:37])
# print(macdsignal[34:37])
# print(macdhist[34:37])
#
rsi = RSI(fechamento, timeperiod=14)
# print('-----------------------------------RSI-----------------------------------')
# print(rsi[34:37])
# Momentum Indicators # Volume Indicators
mfi = MFI(maximo, minimo, fechamento, volume, timeperiod=14)
# print('-----------------------------------MFI-----------------------------------')
# print(mfi[34:37])

#Volume Indicators
ad = AD(maximo, minimo, fechamento, volume)
adosc = ADOSC(maximo, minimo, fechamento, volume, fastperiod=3, slowperiod=10)
obv = OBV(fechamento, volume)

# dataFrameConcatened = pd.concat(fechamento, rsi)
dataFrameConcatened = np.concatenate((abertura, rsi), axis=0)
print(dataFrameConcatened)

ax = plt.gca()
# brsr6CSV.plot(kind='line',x='DATA',y='ABERTURA',ax=ax)
brsr6CSV.plot(kind='line',x='DATA',y='FECHAMENTO', color='red', ax=ax)
plt.plot(upBR, label='BB UP')
plt.plot(lowBR, label='BB Low')
plt.show()
