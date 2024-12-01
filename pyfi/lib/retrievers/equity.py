import pandas as pd
import urllib
from datetime import datetime 
from dateutil.relativedelta import relativedelta

from pandas_datareader import data as pdr
import pandas_datareader.data as web

import yfinance as yf
yf.pdr_override() # !important

START_DATE = datetime.today() - relativedelta(years=1)
END_DATE = datetime.today()

def get_historical_data(tickers, start_date=START_DATE, end_date=END_DATE):
    return pdr.get_data_yahoo(tickers, start_date, end_date, progress=False)#.dropna(how='any', axis=0)

def get_price_matrix(tickers, start_date=START_DATE, end_date=END_DATE):
    return pdr.get_data_yahoo(tickers, start_date, end_date, progress=False)['Close']#.dropna(how='any', axis=0)

def get_price_array(tickers,start_date=START_DATE, end_date=END_DATE):
    return get_price_matrix(tickers, start_date, end_date).to_numpy()

def get_return_matrix(tickers, start_date=START_DATE, end_date=END_DATE):
    return get_price_matrix(tickers, start_date, end_date).pct_change().dropna(how='any', axis=0)

def get_return_array(tickers, start_date=START_DATE, end_date=END_DATE):
    return get_return_matrix(tickers, start_date, end_date).to_numpy()

def get_sp500_constituents():
    url = 'https://www.slickcharts.com/sp500'
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    response = opener.open(url)
    
    return pd.read_html(response.read(), attrs={"class":"table table-hover table-borderless table-sm"})[0].Symbol.tolist()


def get_ff_factors(self, n=3):
    ff_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                                start=self.START_DATE)
    print(ff_dict['DESCR'])
    df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                                        start=self.START_DATE)[0]
    df_three_factor = df_three_factor.div(100)
    df_three_factor.index = df_three_factor.index.strftime('%Y-%m-%d')
    df_three_factor.index.name = 'Date'
    df_three_factor.index = pd.to_datetime(df_three_factor.index)
    return df_three_factor