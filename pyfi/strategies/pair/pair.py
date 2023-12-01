import sys, os
sys.path.append(r"C:\Users\micha\OneDrive\Documents\code\pyfi\\")
import pandas as pd

from pyfi.base.retrievers import equity
from pyfi.core.timeseries import TimeSeries, Frequency, AggFunc
from pyfi.analytics.time_series.machine_learning.regression import RegType

import pandas as pd
from functools import reduce

import numpy as np

class Pair:

    def __init__(self, cls:TimeSeries):

        self.cls = cls

    
    def scan(self):

        r2 = self.cls.correlate(plot = False, permutations=True).rename(
            columns = {'Correlation':'r2','p-value':'r2:p'})[['id', 'r2', 'r2:p']]
        # r2['r2'] = np.round(r2['r2'])
        # r2['r2:p'] = np.round(r2['r2:p'])
        
        ci, cij = self.cls.cointegrate()

        ci.rename(columns={'p_value':'ci:p', 'significance_1%':'ci:alph1','significance_5%':'ci:alph5'},
                    inplace=True)
        ci = ci[['id','ci:alph1','ci:alph5']]
        
        cij.rename(columns={'p_value':'cij:p', 'significance_1%':'cij:alph1','significance_5%':'cij:alph5'},
                    inplace=True)
        cij = cij[['id','cij:alph1','cij:alph5']]
        
        reg_summary, reg_spread, reg_spread_z, reg_spread_adf = self.cls.regress(how = RegType.PERMUTATIONS)

        reg_summary.rename(columns = {'p_value_intercept':'ols:p:int','p_value_coefficient':'ols:p:coef'}, inplace=True)
        reg_summary.drop(['ts1','ts2'], axis=1, inplace = True)

        reg_spread.rename(columns = {'value':'ols:sprd'}, inplace = True)
        reg_spread = reg_spread[reg_spread['date']==reg_spread['date'].max()]
        reg_spread['ols:sprd'] =  reg_spread['ols:sprd'].round(2)
        reg_spread = reg_spread[['id','ols:sprd']]

        reg_spread_z.rename(columns = {'value':'ols:sprd:z'}, inplace = True)
        reg_spread_z = reg_spread_z.sort_values(by='date', ascending = False)
        reg_spread_z = reg_spread_z[reg_spread_z['date']==reg_spread_z['date'].max()]
        reg_spread_z['ols:sprd:z'] =  reg_spread_z['ols:sprd:z'].round(2)
        reg_spread_z = reg_spread_z[['id','ols:sprd:z']]

        reg_spread_adf.rename(columns = {'p-value':'ols:sprd:adf:p'}, inplace = True)
        reg_spread_adf['ols:sprd:adf:p'] = reg_spread_adf['ols:sprd:adf:p']
        reg_spread_adf = reg_spread_adf[['id', 'ols:sprd:adf:p']]

        price_spread, price_spread_z = self.cls.get_price_spread()

        price_spread.rename(columns = {'value':'price:sprd'}, inplace = True)
        price_spread = price_spread.sort_values(by='date', ascending = False)
        price_spread = price_spread[price_spread['date']==price_spread['date'].max()]
        price_spread['price:sprd'] = price_spread['price:sprd']
        price_spread = price_spread[['id', 'price:sprd']]

        price_spread_z.rename(columns = {'value':'price:sprd:z'}, inplace = True)
        price_spread_z = price_spread_z.sort_values(by='date', ascending = False)
        price_spread_z = price_spread_z[price_spread_z['date']==price_spread_z['date'].max()]
        price_spread_z['price:sprd:z'] = price_spread_z['price:sprd:z']
        price_spread_z = price_spread_z[['id', 'price:sprd:z']]

        frames = [r2, ci, cij, reg_summary, reg_spread, reg_spread_z, reg_spread_adf, price_spread, price_spread_z]

        return reduce(lambda x, y: pd.merge(x, y, on = 'id', how='outer'), frames)
