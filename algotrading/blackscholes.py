##Implementacao da precificacao pela tecnica de black scholes.
##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
import pandas as pd

brsr6CSV = pd.read_csv("data/brsr6.csv")['FECHAMENTO']
csna3CSV = pd.read_csv("data/csna3.csv")['FECHAMENTO']
enat3CSV = pd.read_csv("data/enat3.csv")['FECHAMENTO']
ggbr4CSV = pd.read_csv("data/ggbr4.csv")['FECHAMENTO']
itsa4CSV = pd.read_csv("data/itsa4.csv")['FECHAMENTO']
klbn4CSV = pd.read_csv("data/klbn4.csv")['FECHAMENTO']
oibr4CSV = pd.read_csv("data/oibr4.csv")['FECHAMENTO']
petr4CSV = pd.read_csv("data/petr4.csv")['FECHAMENTO']
stbp3CSV = pd.read_csv("data/stbp3.csv")['FECHAMENTO']
tiet4CSV = pd.read_csv("data/tiet4.csv")['FECHAMENTO']
tiet11CSV = pd.read_csv("data/tiet11.csv")['FECHAMENTO']
trpl4CSV = pd.read_csv("data/trpl4.csv")['FECHAMENTO']
vvar3CSV = pd.read_csv("data/vvar3.csv")['FECHAMENTO']


table = pd.concat([brsr6CSV, csna3CSV, enat3CSV, ggbr4CSV, itsa4CSV, klbn4CSV, oibr4CSV, petr4CSV, stbp3CSV, tiet4CSV, tiet11CSV, trpl4CSV, vvar3CSV], axis=1)
# print(x)

table.columns = [col[1] for col in table.columns]
# print(x.head())

plt.figure(figsize=(14, 7))
for c in table.columns.values:
    plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')


# brsr6.merge(csna3, on='DATA')
# brsr6.merge(enat3, on='DATA')
#
# print('()()()()()')
# print(brsr6)

# df = brsr6.set_index('DATA')
# table = df.pivot(columns='ticker')
# # By specifying col[1] in below list comprehension
# # You can select the stock names under multi-level column
# table.columns = [col[1] for col in table.columns]
# table.head()

# brsr6 = {}
# brsr6['preco'] = 23.7
# brsr6['qte'] = 100
#
# csna3 = {}
# csna3['preco'] = 9.95
# csna3['qte'] = 100
#
# enat3 = {}
# enat3['preco'] = 10.89
# enat3['qte'] = 100
#
# ggbr4 = {}
# ggbr4['preco'] = 14.95
# ggbr4['qte'] = 100
#
# itsa4 = {}
# itsa4['preco'] = 12.85
# itsa4['qte'] = 100
#
# klbn4 = {}
# klbn4['preco'] = 2.947
# klbn4['qte'] = 400
#
# oibr4 = {}
# oibr4['preco'] = 1.84
# oibr4['qte'] = 700
#
# petr4 = {}
# petr4['preco'] = 25.79
# petr4['qte'] = 200
#
# stbp3 = {}
# stbp3['preco'] = 6.65
# stbp3['qte'] = 300
#
# tiet11 = {}
# tiet11['preco'] = 12.53
# tiet11['qte'] = 800
#
# tiet4 = {}
# tiet4['preco'] = 24.7
# tiet4['qte'] = 1000
#
# trpl4 = {}
# trpl4['preco'] = 24.54
# trpl4['qte'] = 100
#
# vvar3 = {}
# vvar3['preco'] = 4.83
# vvar3['qte'] = 300
#
# itub4 = {}
# itub4['preco'] = 35.00
# itub4['qte'] = 200


plt.style.use('fivethirtyeight')
np.random.seed(777)

