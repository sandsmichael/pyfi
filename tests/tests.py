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


    # grp by weekly/monthly over trailing 1-yr and describe
    from pyfi.analytics.time_series.stats.descriptive import Descriptive

# test_time_series()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ options                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

def test_options():
  from pyfi.base.retrievers import options
  from pyfi.core.options.options import Chain, Contract, OptionType, OptionExposure

  ticker = 'TLT'
  target_date = '2023-12-31'

  # exp_dates = options.get_expiration_dates(ticker = ticker)
  # target_chain_date = options.closest_date(target_date = target_date, date_list=exp_dates)
  calls, puts = options.get_option_chain(ticker = ticker, date = target_date, strike_bounds=0.05)
  # print(puts)

  # cat_calls, cat_puts = options.concat_option_chain(ticker = ticker)
  # print(cat_puts)

  # Single Contract
  # opt = Contract(contract = puts.iloc[0], opt_type = OptionType.PUT, opt_expo = OptionExposure.LONG,
  #                      underlying_price = 90)
  # opt.process()
  # print(opt.res)
  # print(opt.full_res)

  # Chain given exp date
  optc = Chain(chain = puts, option_type = OptionType.PUT, option_exposure = OptionExposure.SHORT, underlying_price = 90)
  print(optc.processed_chain)
  optc.processed_chain.to_excel('chain.xlsx')

test_options()



def test_option_strategies():
  from pyfi.core.options.strategies.vertical_put_spread import VerticalPutSpread

  VerticalPutSpread().plot()



# test_option_strategies()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ probability analysis                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

def test_probability():
  from pyfi.base.retrievers import equity
  from pyfi.analytics.time_series.stats.probability import Probability
  from pyfi.analytics.time_series.stats.descriptive import Descriptive
  df = equity.get_return_matrix(tickers = ['AMZN', 'AAPL', 'TLT'], start_date='2023-01-01', end_date='2023-11-30') * 100

  prob = Probability(
      df = df,
  )

  # print(prob.standardize())
  print(prob.scenario_probabilites())
  # print(prob.scenario_z_scores())
  print(Descriptive(df=df).describe())


# test_probability()



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ MonteCarlo                                                                                                       │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def test_monte_carlo():
  from pyfi.analytics.time_series.stats.montecarlo import MonteCarlo
  from pyfi.base.retrievers import equity
  df = equity.get_price_matrix(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30') 

  mc = MonteCarlo(prices = df, num_simulations = 1000, n_periods = 252)
  simulated, full_simulation = mc.run()
  mc.plot()
  print(mc.full_simulation)
  print(mc.describe())

# test_monte_carlo()



# """ 
#   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
#   │ technical analyis                                                                                                │
#   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
#  """
# from pyfi.base.retrievers import equity
# from pyfi.core.timeseries import TimeSeries, 

# # df = equity.get_historical_data(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30')
# df = equity.get_price_matrix(tickers = ['AMZN', 'GOOG','GOOGL'], start_date='2023-01-01', end_date='2023-11-30')

# ts = TimeSeries(
#     df = df,
#     dep_var = 'AMZN',
#     indep_var = None
# )