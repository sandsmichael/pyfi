from more_itertools import distinct_combinations as idc
import pandas as pd
import numpy as np

class PriceSpread:

    def __init__(self, df):
        self.df = df

        self.pairs = self.get_combinations()


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


