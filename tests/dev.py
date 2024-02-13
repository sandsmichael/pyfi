import sys, os
sys.path.append(r"C:\Users\micha\OneDrive\Documents\code\pyfi\\")

import pandas as pd
import numpy as np

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ timeseries object     & pairs                                                                                    │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def test_time_series():
    from pyfi.retrievers import equity
    from pyfi.lib.time_series.timeseries import TimeSeries, Frequency, AggFunc
    from pyfi.analytics.time_series.machine_learning.regression import RegType

    # df = equity.get_historical_data(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30')
    df = equity.get_price_matrix(tickers = ['AMZN', 'GOOG','GOOGL'], start_date='2023-01-01', end_date='2023-11-30')

    ts = TimeSeries(
        df = df,
        dep_var = 'AMZN',
        indep_var = None
    )
    # ts.group(frequency = Frequency.DAILY, aggfunc=AggFunc.LAST)
    ts.curate()
    ts.winsorize(subset=None, limits=[0.1, 0.1])
    ts.check_stationarity(alpha=0.05)
    ts.scale()
    ts.unscale()
    ts.decompose(var = ts.dep_var, period = 30, plot=False)
    # pair trade
    ts.correlate(plot = False)
    ts.cointegrate()
    ts.regress(how = RegType.COMBINATIONS)
    ts.get_price_spread()

    # correlation
    # print(ts.correlation_summary)

    # # cointegration
    # print(ts.cointegratation)
    # print(ts.cointegration_johansen)

    # price spread
    # print(ts.price_spread)
    # print(ts.price_spread_z_score)


    # regression
    summary, spread, spread_z, spread_adf = ts.regress(how = RegType.COMBINATIONS)
    print(summary)
    print(spread)
    print(spread_z)
    print(spread_adf)


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

  ticker = 'VXX'
  target_date = '2023-12-31'

  # exp_dates = options.get_expiration_dates(ticker = ticker)
  # target_chain_date = options.closest_date(target_date = target_date, date_list=exp_dates)
  calls, puts = options.get_option_chain(ticker = ticker, date = target_date, strike_bounds=0.05)
  # print(puts)

  # cat_calls, cat_puts = options.concat_option_chain(ticker = ticker)
  # print(cat_puts)

  # Single Contract
  # opt = Contract(contract = puts.iloc[0], opt_type = OptionType.PUT, opt_expo = OptionExposure.LONG,
  #                      spot = 90)
  # opt.process()
  # print(opt.res)
  # print(opt.full_res)

  # Chain given exp date
  optc = Chain(ticker = ticker, chain = puts, option_type = OptionType.CALL, option_exposure = OptionExposure.LONG, spot = 91.5)
  print(optc.processed_chain)
  optc.processed_chain.to_excel('chain.xlsx')

# test_options()



def test_option_strategies():

  from pyfi.core.options.strategies.bull_put_credit_spread import VerticalPutSpread
  from pyfi.base.retrievers import options
  from pyfi.core.options.options import Chain, Contract, OptionType, OptionExposure
  import QuantLib as ql
  from datetime import datetime
  today = datetime.today()

  leg_long = Contract(ticker='TLT', option_type = OptionType.PUT, option_exposure = OptionExposure.SHORT,
                 valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(22, 12, 2023), 
                 premium=4, spot=None, K=90.5, ivol=None) #contract_id=self.contract_id

  leg_short = Contract(ticker='TLT', option_type = OptionType.PUT, option_exposure = OptionExposure.LONG,
                 valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(22, 12, 2023), 
                 premium=2, spot=None, K=85.5, ivol=None) #contract_id=self.contract_id

  vps = VerticalPutSpread(clsLong=leg_long, clsShort=leg_short)
  vps.plot()
  print('max profit:', vps.max_profit)
  print('max loss:', vps.max_loss)
  print('pl ratio:', vps.pl_ratio,  'pl odds', f"{1/vps.pl_ratio}:1")

  

# test_option_strategies()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ options notebook                                                                                                 │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def test_options_notebook():
  import sys, os
  sys.path.append(r"C:\Users\micha\OneDrive\Documents\code\pyfi\\")

  from pyfi.retrievers import equity, options
  from pyfi.analytics.time_series.stats.probability import Probability
  from pyfi.analytics.time_series.stats.descriptive import Descriptive
  from pyfi.core.underlying import Underlying
  from pyfi.lib.time_series.timeseries import TimeSeries, Frequency, AggFunc
  from pyfi.core.options.options import Chain, Contract, OptionType, OptionExposure
  from pyfi.core.options.strategies.bull_put_credit_spread import BullPutCreditSpread
  from pyfi.core.options.strategies.bear_call_credit_spread import BearCallCreditSpread

  import QuantLib as ql
  from datetime import datetime

  import pandas as pd
  import numpy as np
  import seaborn as sns
  import warnings
  warnings.filterwarnings('ignore')

  #https://arch.readthedocs.io/en/latest/univariate/forecasting.html

  TICKER = 'VTWO'
  START_DATE = '2023-09-01'
  TARGET_EXP_DATE = '2023-12-31'

  asset = Underlying(ticker = TICKER, start_date=START_DATE, end_date=None)


  ################ chain  analysis

  # calls, puts = options.get_option_chain(ticker = TICKER, date = TARGET_EXP_DATE, strike_bounds=0.05)
  # print(calls)

  # ch = Chain(ticker = TICKER, chain = puts, option_type = OptionType.PUT, option_exposure = OptionExposure.LONG, spot = None)
  # processed = ch.process_chain()
  # print(processed)

  # all_calls, all_puts = options.concat_option_chain(TICKER)
  # ch = Chain(ticker = TICKER, chain = all_calls)
  # ch.get_volatility_skew(plot=True)

  # print(ch.get_volatility_skew(plot=False))


  ################ individual contract analysis
 
  today = datetime.today()

  # ctr = Contract(ticker='VTWO', option_type = ql.Option.Call, option_exposure = OptionExposure.LONG,
  #               valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(19, 1, 2024), 
  #               premium=2, K=81, ivol=.23) 
  # print(ctr.analyze().T)


  ############## vertical bull put spread
  leg_short = Contract(ticker='EEM', option_type = OptionType.PUT, option_exposure = OptionExposure.SHORT,
                valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(20, 1, 2024), 
                premium=.44, spot=None, K=39, ivol=None) 

  leg_long = Contract(ticker='EEM', option_type = OptionType.PUT, option_exposure = OptionExposure.LONG,
                  valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(20, 1, 2024), 
                  premium=.24, spot=None, K=37.5, ivol=None) 

  vps = BullPutCreditSpread(n_contracts = 1, clsLong=leg_long, clsShort=leg_short)
  vps.plot()


  ############### vertical bear call credit  spread
  # leg_short = Contract(ticker='EEM', option_type = OptionType.CALL, option_exposure = OptionExposure.SHORT,
  #               valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(20, 1, 2024), 
  #               premium=.44, spot=None, K=39, ivol=None) 

  # leg_long = Contract(ticker='EEM', option_type = OptionType.CALL, option_exposure = OptionExposure.LONG,
  #                 valuation=ql.Date(today.day, today.month, today.year), expiration=ql.Date(20, 1, 2024), 
  #                 premium=.24, spot=None, K=37.5, ivol=None) 

  # vps = BearCallCreditSpread(n_contracts = 1, clsLong=leg_long, clsShort=leg_short)
  # vps.plot()



test_options_notebook()









""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ probability analysis                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

def test_probability():
  from pyfi.base.retrievers import equity
  from pyfi.analytics.time_series.stats.probability import Probability
  from pyfi.analytics.time_series.stats.descriptive import Descriptive
  df = equity.get_return_matrix(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30') * 100

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



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ technical analyis                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
# from pyfi.base.retrievers import equity
# from pyfi.core.timeseries import TimeSeries

# df = equity.get_price_matrix(tickers = ['ASML', 'GOOGL'], start_date='2023-01-01', end_date='2023-11-30')
# df = equity.get_historical_data(tickers = ['ASML', 'GOOGL','ASML'], 
#                                 start_date='2023-01-01', 
#                                 end_date='2023-11-30')[['Close']]

# ts = TimeSeries(
#     df = df,
#     dep_var = 'AMZN',
#     indep_var = None
# )

# rsi = ts.rsi()
# # print(rsi)
# bb = ts.bollinger_bands()
# # print(bb)
# dd = ts.max_drawdown(window=30)
# # print(dd)
# macd = ts.macd()
# # print(macd)
# obv = ts.obv()
# print(obv)
# print(ts.get_explained_variance(plot=True))


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ regression                                                                                                       │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def test_simple_regression():
  from pyfi.base.retrievers import equity
  from pyfi.lib.time_series.timeseries import TimeSeries

  df = equity.get_historical_data(tickers = ['SPY', 'ASML', 'GOOGL','AMZN'], 
                                  start_date='2023-01-01', 
                                  end_date='2023-11-30')[['Close']]
  df = df.droplevel(axis=1, level=0).dropna(axis=0, how='any')
  print(df)

  from pyfi.analytics.time_series.stats.inspect import Inspect
  # vif = Inspect(df = df.droplevel(axis=1, level=0)).vif()
  # print(vif)

  from pyfi.analytics.time_series.machine_learning.regression import Regression
  reg = Regression(df=df, dep_var='SPY')
  reg.split()
  reg.fit()
  print(reg.model.summary())
  # reg.test(plot=False)
  # reg.plot_features()
  # reg.plot_resid()
  # reg.plot_qq()

# test_simple_regression()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ garch                                                                                                            │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
def test_garch():
  from pyfi.base.retrievers import equity
  from pyfi.lib.time_series.timeseries import TimeSeries
  from arch import arch_model
  from pyfi.analytics.time_series.autoregressive.garch import Garch
  df = equity.get_return_matrix(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30').multiply(100)
  print(df)
  Garch(rets=df, order=(1,1)).run(forecast_horizon=30)




def test_option_chain_hvol_garch():
  from pyfi.base.retrievers import options
  from pyfi.core.options.options import Chain, Contract, OptionType, OptionExposure

  ticker = 'TLT'
  target_date = '2023-12-31'

  calls, puts = options.get_option_chain(ticker = ticker, date = target_date, strike_bounds=0.05)
  optc = Chain(ticker = ticker, chain = puts, option_type = OptionType.PUT, option_exposure = OptionExposure.SHORT, spot = 94.5)
  optc.processed_chain
  # optc.processed_chain.to_excel('chain.xlsx')


def test_underlying():
  from pyfi.core.underlying import Underlying

  asset = Underlying(ticker = 'VTWO')

  # asset.plot_garch()
  asset.add_bollinger_bands()
  asset.add_rsi()
  asset.add_macd()
  asset.add_on_balance_volume()
  print(asset.df)

# test_underlying()



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ feature engine                                                                                                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

# def test_feature_engine():
#   from pyfi.base.retrievers import fred 
#   from pyfi.analytics.feature_engine.feature_engine import FeatureEngine
#   from pyfi.base.retrievers import equity
#   from pyfi.analytics.time_series.machine_learning.regression import RegType

#   # retreive data
#   start_date = '2020-01-01' 
#   end_date = '2023-12-07' 

#   df_a = equity.get_price_matrix(tickers = [ 'TLT', 'AMZN', 'QQQ', 'BTC-USD'], start_date=start_date, end_date=end_date)

#   df_b = fred.get_fred_data(['VIXCLS', 'SP500', 'DGS10', 'SOFR', ] , start_date, end_date)

#   df = df_a.merge(df_b, left_index = True, right_index = True)

#   print(df.tail())

#   # instantiate feature engine
#   eng = FeatureEngine(df = df, dep_var = 'SP500')
#   print(eng.get_explained_variance())
#   print(eng.vif())
#   # print(eng.check_stationarity())
#   # print(eng.decompose(var = eng.dep_var, period = 30, plot=False))
#   # print(eng.correlate(plot = False))
#   # print(eng.cointegrate())
#   # print(eng.regress(how = RegType.COMBINATIONS))
#   # print(eng.get_price_spread())
#   # print(eng.correlation_feature_selection())
#   # print(eng.rfe_feature_selection())
#   # print(eng.select_k_best_feature_selection())
#   # print(eng.lasso_feature_selection())
#   # print(eng.tree_based_feature_importance())
#   # print(eng.rfa_feature_selection())
#   print(eng.create_polynomial_features())
# test_feature_engine()