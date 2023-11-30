# https://medium.com/analytics-vidhya/monte-carlo-simulations-for-predicting-stock-prices-python-a64f53585662
# https://www.investopedia.com/articles/07/montecarlo.asp

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf

data =  yf.download("AMZN", start="2010-01-01", end="2022-10-30")['Adj Close']
data.plot(figsize=(15,6))
plt.show()

log_returns = np.log(1 + data.pct_change())
#Plot
sns.distplot(log_returns.iloc[1:])
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

# drift
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)


#variance and daily ret
stdev = log_returns.std()
days = 50
trials = 100
Z = norm.ppf(np.random.rand(days, trials)) #days, trials
daily_returns = np.exp(drift + stdev * Z)


# forecast stock price 
price_paths = np.zeros_like(daily_returns)
price_paths[0] = data.iloc[-1]
for t in range(1, days):
    price_paths[t] = price_paths[t-1]*daily_returns[t]

df = pd.DataFrame(price_paths)
print(df)
df.plot()
plt.show()