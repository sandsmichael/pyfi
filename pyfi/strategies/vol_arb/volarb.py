
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
#                      spot = 90)
# opt.process()
# print(opt.res)
# print(opt.full_res)

# Chain given exp date
optc = Chain(ticker = ticker, chain = puts, option_type = OptionType.PUT, option_exposure = OptionExposure.SHORT, spot = 91.5)
print(optc.processed_chain)
optc.processed_chain.to_excel('chain.xlsx')




# from pyfi.base.retrievers import equity
# from pyfi.core.timeseries import TimeSeries
# from arch import arch_model
# from pyfi.analytics.time_series.autoregressive.garch import Garch
# df = equity.get_return_matrix(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30').multiply(100)
# print(df)
# Garch(rets=df, order=(1,1)).run(forecast_horizon=30)