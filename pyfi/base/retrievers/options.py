import pandas as pd
import urllib
from datetime import datetime 

from pandas_datareader import data as pdr
import yfinance as yf
from yahoo_fin import options


yf.pdr_override() # !important


def get_expiration_dates(ticker):
    return [pd.to_datetime(d,format = '%B %d, %Y').strftime('%Y-%m-%d') for d in options.get_expiration_dates(ticker)]


def closest_date(target_date, date_list):
    target_date = datetime.strptime(target_date, '%Y-%m-%d') 
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]
    time_diff = [abs(target_date - date) for date in date_objects]
    min_index = time_diff.index(min(time_diff))
    return date_list[min_index]


def extract_expiration_date(df, how = None):
    if how == 'Call':
        df['Expiration Date'] = df['Contract Name'].str.rsplit('C').str[0].str[-6:]
    elif how == 'Put':
        df['Expiration Date'] = df['Contract Name'].str.rsplit('P').str[0].str[-6:]
    df['Expiration Date'] = [pd.to_datetime(x, format="%y%m%d").strftime("%Y-%m-%d") for x in df['Expiration Date']]
    return df


def process_pricing_model_inputs(df):
    """ Prep BSM inputs to solve for IV for NPV
    """
    df['Expiration_dt'] = pd.to_datetime(df['Expiration Date'])
    df['Spot'] = 91
    # df['_Strike'] = df['Strike']
    df['Market_IV'] = pd.to_numeric(df['Implied Volatility'].str.replace('%','')) / 100
    df['Dividend'] = 0
    df['Rfr'] = 0.05
    df['HistoricalVol'] = 0

    return df


def get_option_chain(ticker, date):
    """ Retreives option chain for an expiration date. If invalid date is provided, closest expiration is returned.
    """
    chain_date = closest_date(target_date=date, date_list = get_expiration_dates(ticker=ticker))
    chain = options.get_options_chain(ticker, chain_date)
    calls, puts = chain['calls'], chain['puts']

    calls = process_pricing_model_inputs(extract_expiration_date(calls, how = 'Call'))
    puts = process_pricing_model_inputs(extract_expiration_date(puts, how = 'Put'))

    return calls, puts


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
# def get_full_chain(ticker):
