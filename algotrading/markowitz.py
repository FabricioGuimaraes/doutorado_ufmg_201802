##Validacao de uma carteira pela Teoria de Markowitz
##
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

brsr6CSV = pd.read_csv("data/brsr6.csv", usecols=['DATA', 'FECHAMENTO'])
brsr6CSV['FECHAMENTO'] = brsr6CSV['FECHAMENTO'].str.replace(',', '.')
brsr6CSV['FECHAMENTO'] = brsr6CSV['FECHAMENTO'].astype(float)
brsr6CSV['ticker'] = 'brsr6'
csna3CSV = pd.read_csv("data/csna3.csv", usecols=['DATA', 'FECHAMENTO'])
csna3CSV['FECHAMENTO'] = csna3CSV['FECHAMENTO'].str.replace(',', '.')
csna3CSV['FECHAMENTO'] = csna3CSV['FECHAMENTO'].astype(float)
csna3CSV['ticker'] = 'csna3'
enat3CSV = pd.read_csv("data/enat3.csv", usecols=['DATA', 'FECHAMENTO'])
enat3CSV['FECHAMENTO'] = enat3CSV['FECHAMENTO'].str.replace(',', '.')
enat3CSV['FECHAMENTO'] = enat3CSV['FECHAMENTO'].astype(float)
enat3CSV['ticker'] = 'enat3'
ggbr4CSV = pd.read_csv("data/ggbr4.csv", usecols=['DATA', 'FECHAMENTO'])
ggbr4CSV['FECHAMENTO'] = ggbr4CSV['FECHAMENTO'].str.replace(',', '.')
ggbr4CSV['FECHAMENTO'] = ggbr4CSV['FECHAMENTO'].astype(float)
ggbr4CSV['ticker'] = 'ggbr4'
itsa4CSV = pd.read_csv("data/itsa4.csv", usecols=['DATA', 'FECHAMENTO'])
itsa4CSV['FECHAMENTO'] = itsa4CSV['FECHAMENTO'].str.replace(',', '.')
itsa4CSV['FECHAMENTO'] = itsa4CSV['FECHAMENTO'].astype(float)
itsa4CSV['ticker'] = 'itsa4'
klbn4CSV = pd.read_csv("data/klbn4.csv", usecols=['DATA', 'FECHAMENTO'])
klbn4CSV['FECHAMENTO'] = klbn4CSV['FECHAMENTO'].str.replace(',', '.')
klbn4CSV['FECHAMENTO'] = klbn4CSV['FECHAMENTO'].astype(float)
klbn4CSV['ticker'] = 'klbn4'
oibr4CSV = pd.read_csv("data/oibr4.csv", usecols=['DATA', 'FECHAMENTO'])
oibr4CSV['FECHAMENTO'] = oibr4CSV['FECHAMENTO'].str.replace(',', '.')
oibr4CSV['FECHAMENTO'] = oibr4CSV['FECHAMENTO'].astype(float)
oibr4CSV['ticker'] = 'oibr4'
petr4CSV = pd.read_csv("data/petr4.csv", usecols=['DATA', 'FECHAMENTO'])
petr4CSV['FECHAMENTO'] = petr4CSV['FECHAMENTO'].str.replace(',', '.')
petr4CSV['FECHAMENTO'] = petr4CSV['FECHAMENTO'].astype(float)
petr4CSV['ticker'] = 'petr4'
stbp3CSV = pd.read_csv("data/stbp3.csv", usecols=['DATA', 'FECHAMENTO'])
stbp3CSV['FECHAMENTO'] = stbp3CSV['FECHAMENTO'].str.replace(',', '.')
stbp3CSV['FECHAMENTO'] = stbp3CSV['FECHAMENTO'].astype(float)
stbp3CSV['ticker'] = 'stbp3'
tiet4CSV = pd.read_csv("data/tiet4.csv", usecols=['DATA', 'FECHAMENTO'])
tiet4CSV['FECHAMENTO'] = tiet4CSV['FECHAMENTO'].str.replace(',', '.')
tiet4CSV['FECHAMENTO'] = tiet4CSV['FECHAMENTO'].astype(float)
tiet4CSV['ticker'] = 'tiet4'
tiet11CSV = pd.read_csv("data/tiet11.csv", usecols=['DATA', 'FECHAMENTO'])
tiet11CSV['FECHAMENTO'] = tiet11CSV['FECHAMENTO'].str.replace(',', '.')
tiet11CSV['FECHAMENTO'] = tiet11CSV['FECHAMENTO'].astype(float)
tiet11CSV['ticker'] = 'tiet11'
trpl4CSV = pd.read_csv("data/trpl4.csv", usecols=['DATA', 'FECHAMENTO'])
trpl4CSV['FECHAMENTO'] = trpl4CSV['FECHAMENTO'].str.replace(',', '.')
trpl4CSV['FECHAMENTO'] = trpl4CSV['FECHAMENTO'].astype(float)
trpl4CSV['ticker'] = 'trpl4'
vvar3CSV = pd.read_csv("data/vvar3.csv", usecols=['DATA', 'FECHAMENTO'])
vvar3CSV['FECHAMENTO'] = vvar3CSV['FECHAMENTO'].str.replace(',', '.')
vvar3CSV['FECHAMENTO'] = vvar3CSV['FECHAMENTO'].astype(float)
vvar3CSV['ticker'] = 'vvar3'

selected = ['brsr6CSV', 'csna3CSV', 'enat3CSV', 'ggbr4CSV', 'itsa4CSV', 'klbn4CSV', 'oibr4CSV', 'petr4CSV', 'stbp3CSV', 'tiet4CSV', 'tiet11CSV', 'trpl4CSV', 'vvar3CSV']
frames = [brsr6CSV, csna3CSV, enat3CSV, ggbr4CSV, itsa4CSV, klbn4CSV, oibr4CSV, petr4CSV, stbp3CSV, tiet4CSV, tiet11CSV, trpl4CSV, vvar3CSV]
frames_merged = pd.concat(frames).set_index('DATA')
frames_merged = frames_merged.pivot(columns='ticker')

# calculate daily and annual returns of the stocks
returns_daily = frames_merged.pct_change()
returns_annual = returns_daily.mean() * 250

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)
# num_assets = len(frames)
num_portfolios = 50000

# #set random seed for reproduction's sake
# np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(selected):
# for counter,symbol in enumerate(frames):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]
# column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in frames]
# reorder dataframe columns
df = df[column_order]

# # plot frontier, max sharpe & min Volatility values with a scatterplot
# plt.style.use('seaborn-dark')
# df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
#                 cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
# plt.xlabel('Volatility (Std. Deviation)')
# plt.ylabel('Expected Returns')
# plt.title('Efficient Frontier')
# plt.show()

# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

# print the details of the 2 special portfolios
print('Wallet composition Weights')
print(min_variance_port.T)
print(sharpe_portfolio.T)
print()