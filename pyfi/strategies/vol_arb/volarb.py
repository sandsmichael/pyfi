
from pyfi.base.retrievers import equity
from pyfi.core.timeseries import TimeSeries
from arch import arch_model
from pyfi.analytics.time_series.autoregressive.garch import Garch
df = equity.get_return_matrix(tickers = ['AMZN'], start_date='2023-01-01', end_date='2023-11-30').multiply(100)
print(df)
Garch(rets=df, order=(1,1)).run(forecast_horizon=30)