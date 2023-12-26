from pyfi.core.timeseries import TimeSeries
from pyfi.analytics.time_series.technical_analysis.ta import TechnicalAnalysis
from datetime import datetime 
import numpy as np
from pyfi.retrievers import equity
from pyfi.analytics.time_series.autoregressive.garch import Garch
# import ta
today = datetime.today().strftime('%Y-%m-%d')


class Underlying(TechnicalAnalysis):
    """ An underlying asset 
    """

    def __init__(
        self,
        ticker:str = None,
        period:int = 30,
        start_date = '2023-01-01',
        end_date = None
    ) -> None:
        
        self.ticker = ticker

        self.period = period

        if end_date is None:
            end_date = today

        self.df = equity.get_historical_data(tickers=[ticker], start_date=start_date, end_date=end_date)
        
        self.price_ts = self.df['Close']
        
        super().__init__(df = self.df)
        

    @property
    def spot(self):
        return self.price_ts.iloc[-1]
    
    @spot.setter
    def spot(self, value):
        self._spot = value

    @property
    def stdev(self):
        return self.price_ts.std() 

    @property
    def hvol(self):
        # Historical observed volatility: Average standard deviation of returns based on a 30 day rolling window over the start and end date period; annualized.
        return self.price_ts.pct_change().rolling(window=30).std().mean() * np.sqrt(252)  # .quantile(.75)

    @property  
    def hvol_two_sigma(self): 
        return self.hvol*2

    @property  
    def hvol_garch(self):   
        return Garch(rets = self.price_ts.pct_change().multiply(100).dropna(axis=0, how='any'), 
                    order = (1,1)).forecast(forecast_horizon=self.period)


    def fit_garch(self, order:tuple = (1,1), plot=False):

        g = Garch(rets = self.price_ts.pct_change().multiply(100).dropna(axis=0, how='any'), 
                    order = order)
        
        results = g.fit()
        
        if plot:
            g.plot(results)

        return results

    