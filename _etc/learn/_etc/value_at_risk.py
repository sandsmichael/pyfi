import numpy as np
from datetime import datetime
import yfinance as yf
import pandas_datareader.data as pdr
import pandas as pd

import scipy.stats as stats

yf.pdr_override()

ticker = 'GOOG'
start = datetime.strptime('2018-01-01', '%Y-%m-%d')
end = datetime.strptime('2022-12-10', '%Y-%m-%d')

historical_data = pdr.DataReader(ticker, start, end , data_source='yahoo')

alpha = 0.01 # confidence interval to estimate with
num_shares = 1
on_date = '2021-03-09'
share_price = historical_data['Adj Close'][on_date]

# Use z value, mean and standard deviation; Assume return is normal distribution
Z_value = stats.norm.ppf(abs(alpha))
mean_return_rate = historical_data['Adj Close'].pct_change().mean()
std_return_rate = historical_data['Adj Close'].pct_change().std()

rel_VaR = -num_shares * share_price * Z_value * std_return_rate 
abs_VaR = -num_shares * share_price * (Z_value * std_return_rate + mean_return_rate)

print("The estimated relative VaR and absolute VaR of an investment of", num_shares, "shares of", ticker, "on", on_date, "with price $", round(share_price,2), "per share is $", round(rel_VaR,2), "and $", round(abs_VaR,2), "respectively.")