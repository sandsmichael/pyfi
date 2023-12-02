import sys, os
sys.path.append(r"C:\Users\micha\OneDrive\Documents\code\pyfi\\")
import pandas as pd

from pyfi.base.retrievers import equity
from pyfi.core.timeseries import TimeSeries, Frequency, AggFunc
from pyfi.analytics.time_series.machine_learning.regression import RegType

import pandas as pd
from functools import reduce

import numpy as np

import warnings
warnings.filterwarnings('ignore')


class Pair:

    def __init__(self, cls:TimeSeries):

        self.cls = cls


    def correlate(self):
        r2 = self.cls.correlate(plot = False, permutations=True).rename(columns = {'Correlation':'r2','p-value':'r2:p'})[['id', 'r2', 'r2:p']]
        
        r2['r2'] = pd.to_numeric(r2['r2']).round(2)
        
        r2['r2:p'] = pd.to_numeric(r2['r2:p']).round(2)
        
        return r2
        

    def cointegrate(self):
        ci, cij = self.cls.cointegrate()

        ci.rename(columns={'p_value':'ci:p', 'significance_1%':'ci:alph1','significance_5%':'ci:alph5'},
                    inplace=True)
        
        ci = ci[['id','ci:alph1','ci:alph5']]
        cij.rename(columns={'p_value':'cij:p', 'significance_1%':'cij:alph1','significance_5%':'cij:alph5'},
                    inplace=True)
        
        cij = cij[['id','cij:alph1','cij:alph5']]

        return ci, cij
    

    def regress(self, how = RegType.PERMUTATIONS):
        reg_summary, reg_spread, reg_spread_z, reg_spread_adf = self.cls.regress(how = how)

        reg_summary.rename(columns = {'p_value_intercept':'ols:p:int','p_value_coefficient':'ols:p:coef'}, inplace=True)
        reg_summary.drop(['ts1','ts2','ols:p:int'], axis=1, inplace = True)

        reg_spread.rename(columns = {'value':'ols:spd'}, inplace = True)
        reg_spread_ts = reg_spread.copy()
        reg_spread = reg_spread[reg_spread['date']==reg_spread['date'].max()]
        reg_spread['ols:spd'] =  reg_spread['ols:spd'].round(2)
        reg_spread = reg_spread[['id','ols:spd']]

        reg_spread_z.rename(columns = {'value':'ols:spd:z'}, inplace = True)
        reg_spread_z_ts = reg_spread_z.copy()
        reg_spread_z = reg_spread_z[reg_spread_z['date']==reg_spread_z['date'].max()]
        reg_spread_z['ols:spd:z'] =  reg_spread_z['ols:spd:z'].round(2)
        reg_spread_z = reg_spread_z[['id','ols:spd:z']]

        reg_spread_adf.rename(columns = {'p-value':'ols:spd:adf:p'}, inplace = True)
        reg_spread_adf['ols:spd:adf:p'] = pd.to_numeric(reg_spread_adf['ols:spd:adf:p']).round(2)
        reg_spread_adf = reg_spread_adf[['id', 'ols:spd:adf:p']]

        return reg_summary, reg_spread, reg_spread_z, reg_spread_adf, reg_spread_ts, reg_spread_z_ts


    def price_spread(self, how = RegType.PERMUTATIONS):
        price_spread, price_spread_z = self.cls.get_price_spread(how=how)

        # price spread
        price_spread.rename(columns = {'value':'price:spd'}, inplace = True)

        price_spread_ts = price_spread.copy()

        # price_spread = price_spread.sort_values(by='date', ascending = False)
        price_spread = price_spread[price_spread['date']==price_spread['date'].max()]

        price_spread['price:spd'] = pd.to_numeric(price_spread['price:spd']).round(2)
        
        price_spread = price_spread[['id', 'price:spd']]

        # price spread z
        price_spread_z.rename(columns = {'value':'price:spd:z'}, inplace = True)

        price_spread_z_ts = price_spread_z.copy()
        # price_spread_z = price_spread_z.sort_values(by='date', ascending = False)
        price_spread_z = price_spread_z[price_spread_z['date']==price_spread_z['date'].max()]

        price_spread_z['price:spd:z'] = pd.to_numeric(price_spread_z['price:spd:z']).round(2)
        
        price_spread_z = price_spread_z[['id', 'price:spd:z']]

        return price_spread, price_spread_z, price_spread_ts, price_spread_z_ts


    def scan(self):

        r2 = self.correlate()
        ci,cij = self.cointegrate()
        reg_summary, reg_spread, reg_spread_z, reg_spread_adf, reg_spread_ts, reg_spread_z_ts = self.regress()
        price_spread, price_spread_z, price_spread_ts, price_spread_z_ts = self.price_spread()
        
        frames = [r2, ci, cij, reg_summary, reg_spread, reg_spread_z, reg_spread_adf, price_spread, price_spread_z]

        return reduce(lambda x, y: pd.merge(x, y, on = 'id', how='outer'), frames)
