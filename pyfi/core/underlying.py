from pyfi.core.timeseries import TimeSeries
from datetime import datetime 
from pyfi.base.retrievers import equity

# import yfinance as yf
# from yahoo_fin import options
# yf.pdr_override() # !important

today = datetime.today().strftime('%Y-%m-%d')

class Underlying:
    """ An underlying asset 
    """

    def __init__(
        self,
        ticker:str = None
    
    ) -> None:
        
        self.price_ts = equity.get_historical_data(tickers=[ticker], start_date='2023-01-01', end_date=today)['Close']
        
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
