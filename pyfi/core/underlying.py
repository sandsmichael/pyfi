from pyfi.core.timeseries import TimeSeries
from pyfi.analytics.time_series.technical_analysis.ta import TechnicalAnalysis
from datetime import datetime 
from pyfi.base.retrievers import equity
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
    def hvol(self):
        return self.price_ts.std() / self.price_ts.iloc[-1]

    @property  
    def hvol_two_sigma(self): 
        return (self.price_ts.std() * 2) / self.price_ts.iloc[-1]

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

    