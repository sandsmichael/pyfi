import pandas as pd
import numpy as np
from more_itertools import distinct_permutations as idp
from more_itertools import distinct_combinations as idc
from enum import Enum
import statsmodels.api as sm
from pyfi.analytics.time_series.stats.inspect import Inspect

class RegType(Enum):
    UNIVARIATE = 1
    MULTIVARIATE = 2
    PERMUTATIONS = 3
    COMBINATIONS = 4


@staticmethod
def fit(df, row):
    x = df[row['ts1']]
    y = df[row['ts2']]
    
    x_with_const = sm.add_constant(x.to_numpy().reshape(-1, 1))

    model = sm.OLS(y.to_numpy().reshape(-1, 1), x_with_const).fit()

    alpha = model.params[0]
    beta = model.params[1]
    
    spread = y - (alpha + beta * x)

    f_statistic = model.fvalue
    r_squared = model.rsquared
    p_values = model.pvalues

    return (alpha, beta, spread, f_statistic, r_squared, p_values)


class RegressionPairs:

    def __init__(self, cls, how):
        
        self.cls = cls
        
        self.df = self.cls.df
        self.how = how

        if self.how == RegType.PERMUTATIONS:
            self.pairs = self.get_permutations()
        
        elif self.how == RegType.COMBINATIONS:
            self.pairs = self.get_combinations()


    def get_combinations(self):
        return pd.DataFrame(idc(self.df.columns, 2), columns=['ts1', 'ts2'])

    def get_permutations(self):
        return pd.DataFrame(idp(self.df.columns, 2), columns=['ts1', 'ts2'])


    def run(self):
        return self.pairs.apply(lambda row : fit(self.df, row), axis=1)
            

    def get_summary(self):
        """ Fits linear regression model according to `how` param and returns summary statistics. 
        Test regression spread for stationarity.
        """
        reg_results = self.run()

        df_permut = self.pairs.copy()

        # regression results
        df_permut['alpha'] = reg_results.apply(lambda x: x[0])
        df_permut['beta'] = reg_results.apply(lambda x: x[1])
        df_permut['f_statistic'] = reg_results.apply(lambda x: x[3])
        df_permut['r_squared'] = reg_results.apply(lambda x: x[4])
        df_permut[['p_value_intercept', 'p_value_coefficient']] = reg_results.apply(lambda x: pd.Series(x[5]))

        # regression spread
        spread = self.pairs.merge(reg_results.apply(lambda x: x[2]), how = 'outer', left_index = True, right_index = True)
        spread['pair'] = spread.ts1 + '_' + spread.ts2
        spread.drop(columns = ['ts1','ts2'], inplace = True)
        spread.set_index('pair', inplace = True)
        spread = spread.T

        # regression z score
        spread_z_score = (spread - spread.mean()) / spread.std()

        # regression adf stationarity test
        spread_adf = Inspect(df = spread).check_stationarity(alpha = 0.05)
        
        return df_permut, spread, spread_z_score, spread_adf