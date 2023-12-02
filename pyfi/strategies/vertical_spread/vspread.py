

# analysis of the underlying
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


from pyfi.analytics.time_series.stats.montecarlo import MonteCarlo
from pyfi.base.retrievers import equity
df = equity.get_price_matrix(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30') 

mc = MonteCarlo(prices = df, num_simulations = 1000, n_periods = 252)
simulated, full_simulation = mc.run()
mc.plot()
print(mc.full_simulation)
print(mc.describe())


from pyfi.base.retrievers import equity
from pyfi.core.timeseries import TimeSeries

df = equity.get_price_matrix(tickers = ['ASML', 'GOOGL'], start_date='2023-01-01', end_date='2023-11-30')
df = equity.get_historical_data(tickers = ['ASML', 'GOOGL','ASML'], 
                                start_date='2023-01-01', 
                                end_date='2023-11-30')[['Close']]

ts = TimeSeries(
    df = df,
    dep_var = 'AMZN',
    indep_var = None
)

rsi = ts.rsi()
# print(rsi)
bb = ts.bollinger_bands()
# print(bb)
dd = ts.max_drawdown(window=30)
# print(dd)
macd = ts.macd()
# print(macd)
obv = ts.obv()
print(obv)
print(ts.get_explained_variance(plot=True))


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

# test_options()


from pyfi.core.options.strategies.vertical_put_spread import VerticalPutSpread
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

