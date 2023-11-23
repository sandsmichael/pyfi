import pandas as pd
import urllib
from datetime import datetime 

from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override() # !important

def get_historical_data(tickers, start_date, end_date):
    return pdr.get_data_yahoo('AMZN', start_date, end_date).dropna(how='any', axis=1)


def get_sp500_constituents():
    url = 'https://www.slickcharts.com/sp500'
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    response = opener.open(url)
    
    return pd.read_html(response.read(), attrs={"class":"table table-hover table-borderless table-sm"})[0].Symbol.tolist()