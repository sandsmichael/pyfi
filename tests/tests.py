import sys, os
sys.path.append(r"C:\Users\micha\OneDrive\Documents\code\pyfi\\")

import pandas as pd
import numpy as np

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ timeseries object                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def test_time_series():
    from pyfi.base.retrievers import equity
    from pyfi.core.timeseries import TimeSeries, Frequency, AggFunc
    from pyfi.analytics.time_series.machine_learning.regression import RegType

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
    ts.regress(how = RegType.COMBINATIONS)
    ts.get_price_spread()

    # inspect
    print(ts.decomposed)
    print(ts.adf)

    # correlation
    print(ts.correlation_summary)

    # cointegration
    print(ts.cointegratation)
    print(ts.cointegration_johansen)

    # regression
    print(ts.df.tail())
    print(ts.regression_summary)
    print(ts.regression_spread)
    print(ts.regression_spread_z_score)
    print(ts.regression_spread_adf)

    # price spread
    print(ts.price_spread)
    print(ts.price_spread_z_score)

# test_time_series()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ options                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
from pyfi.base.retrievers import options
from pyfi.core.option_chain import OptionChain

ticker = 'TLT'
target_date = '2023-12-31'

# exp_dates = options.get_expiration_dates(ticker = ticker)
# target_chain_date = options.closest_date(target_date = target_date, date_list=exp_dates)
calls, puts = options.get_option_chain(ticker = ticker, date = '2023-12-20')
# print(puts)

# cat_calls, cat_puts = options.concat_option_chain(ticker = ticker)
# print(cat_puts)

opt = OptionChain(chain = puts)
opt.solve_for_npv()

