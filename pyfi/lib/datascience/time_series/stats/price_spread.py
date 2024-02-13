from more_itertools import distinct_combinations as idc
from more_itertools import distinct_permutations as idp
from pyfi.analytics.time_series.machine_learning.regression import RegType
from pyfi.analytics.time_series.stats.inspect import Inspect
import pandas as pd
import numpy as np

class PriceSpread:

    def __init__(self, df, how=RegType.PERMUTATIONS):
        self.df = df

        if how == RegType.PERMUTATIONS:
            self.pairs = self.get_permutations()
        elif how == RegType.COMBINATIONS:
            self.pairs = self.get_combinations()

    def get_permutations(self):
        return pd.DataFrame(idp(self.df.columns, 2), columns=['ts1', 'ts2'])

    def get_combinations(self):
        return pd.DataFrame(idc(self.df.columns, 2), columns=['ts1', 'ts2'])

    @staticmethod
    def calculate_price_ratio(df, row):
        return df[row['ts1']] / df[row['ts2']]

    def get_price_spread(self):
        spread = self.pairs.apply( lambda row : self.calculate_price_ratio(self.df, row), axis=1)
        spread = self.pairs.merge(spread, how = 'outer', left_index = True, right_index = True)
        spread['pair'] = spread.ts1 + '_' + spread.ts2
        spread.drop(columns = ['ts1','ts2'], inplace = True)
        spread.set_index('pair', inplace = True)
        spread.index.name = None
        self.spread = spread.T
        return self.spread
    
    def get_price_spread_z_score(self):
        spread = self.get_price_spread()
        self.spread_z_score = (spread - spread.mean()) / spread.std()
        return self.spread_z_score

    def get_price_spread_adf(self):
        spread = self.get_price_spread()
        return Inspect(df = spread).check_stationarity()

