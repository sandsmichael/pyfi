import sys, os
sys.path.append(r"C:\Users\micha\OneDrive\Documents\code\pyfi\\")

import pandas as pd
import numpy as np

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ timeseries object                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
from pyfi.base.retrievers import equity
from pyfi.core.timeseries import TimeSeries, Frequency, AggFunc

# df = equity.get_historical_data(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30')
df = equity.get_price_matrix(tickers = ['AMZN', 'GOOG','GOOGL'], start_date='2023-01-01', end_date='2023-11-30')

ts = TimeSeries(
    df = df,
    dep_var = 'AMZN',
    indep_var = None
)
ts.group(frequency = Frequency.DAILY, aggfunc=AggFunc.LAST)
ts.curate()
ts.winsorize(subset=None, limits=[0.1, 0.1])
ts.check_stationarity(alpha=0.05)
ts.scale()
ts.unscale()
ts.decompose(var = ts.dep_var, period = 30, plot=False)
ts.correlate(plot = False)
ts.cointegrate()
ts.regress()

## inspect
# print(ts.decomposed)
# print(ts.adf)

## correlation
# print(ts.correlation)
# print(ts.pearson_p_values)
# print(ts.correlation_summary)

## cointegration
# print(ts.cointegratation)
# print(ts.cointegration_johansen)





