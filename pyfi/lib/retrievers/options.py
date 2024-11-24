import pandas as pd
import numpy as np
import urllib
import re
from datetime import datetime, timedelta
from pyfi.lib.retrievers import equity

from pandas_datareader import data as pdr
import yfinance as yf
from yahoo_fin import options as yf_options
yf.pdr_override() # !important


today = datetime.today().strftime('%Y-%m-%d')

def get_next_friday():
    today = datetime.today()
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:  
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday)
    return next_friday.date().strftime("%Y-%m-%d")

def get_expiration_dates(ticker):
    dates = yf_options.get_expiration_dates(ticker)
    if len(dates) > 1:
        res = [pd.to_datetime(d, format = '%B %d, %Y').strftime('%Y-%m-%d') for d in dates]
    else:
        res = [get_next_friday()] # Use the next Fridays date
    return res


def closest_date(target_date, date_list):
    print(date_list)
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d') 
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]
    time_diff = [abs(target_date - date) for date in date_objects]
    min_index = time_diff.index(min(time_diff))
    return date_list[min_index]


def extract_expiration_date(df, how = None):
    # if how == 'Call':
    #     df['Expiration Date'] = df['Contract Name'].str.rsplit('C').str[0].str[-6:]
    # elif how == 'Put':
    #     df['Expiration Date'] = df['Contract Name'].str.rsplit('P').str[0].str[-6:]
    def match_expiration_date(s):
        pattern = r'(\d{2})(\d{2})(\d{2})([PC])'  # Match the date format and P/C
        match = re.search(pattern, s)
        if match:
            year = int(match.group(1)) + 2000  # Convert to four-digit year (assumes 21st century)
            month = int(match.group(2))
            day = int(match.group(3))
            try:
                return datetime(year, month, day).date()
            except Exception as e:
                return np.nan
        return None
    df['Expiration Date'] = df['Contract Name'].apply(match_expiration_date)
    return df


def process_pricing_model_inputs(df, ticker):
    """ Prep BSM inputs to solve for IV for NPV
    """
    df['Expiration_dt'] = pd.to_datetime(df['Expiration Date'])
    df['Market_IV'] = pd.to_numeric(df['Implied Volatility'].str.replace('%','')) / 100
    return df


def get_option_chain(ticker:str, date, strike_bounds:float = None):
    """ Retreives option chain for an expiration date. If invalid date is provided, closest expiration is returned.

    strike_bounds: Percentage above or below market price used to determine the strike prices included in the result set.
    """
    if not isinstance(ticker, str):
        raise ValueError('Ticker should be specified as a string.')
    
    price_ts = equity.get_historical_data(tickers=[ticker], start_date=datetime.today() - timedelta(days=1), end_date=today)['Close'].iloc[-1]
    
    if strike_bounds is not None:
        upper_bound_strike = price_ts * (1 + strike_bounds)
        lower_bound_strike = price_ts * (1 - strike_bounds)

    chain_date = closest_date(target_date=date, date_list = get_expiration_dates(ticker=ticker))
    chain = yf_options.get_options_chain(ticker, chain_date)
    calls, puts = chain['calls'], chain['puts']

    calls = process_pricing_model_inputs(extract_expiration_date(calls, how = 'Call'), ticker = ticker)
    puts = process_pricing_model_inputs(extract_expiration_date(puts, how = 'Put'), ticker = ticker)

    if strike_bounds is not None:
        calls = calls[(calls.Strike >= lower_bound_strike ) & (calls.Strike <= upper_bound_strike)]
        puts = puts[(puts.Strike >= lower_bound_strike ) & (puts.Strike <= upper_bound_strike)]

    return calls.reset_index(drop=True), puts.reset_index(drop=True)


def concat_option_chain(ticker):
    all_calls, all_puts = [], []

    date_list = get_expiration_dates(ticker=ticker)

    for date in date_list:
        calls, puts = get_option_chain(ticker, date)
        all_calls.append(calls)
        all_puts.append(puts)
    
    cat_calls = pd.concat(all_calls, axis=0)
    cat_puts = pd.concat(all_puts, axis=0)

    return cat_calls, cat_puts

