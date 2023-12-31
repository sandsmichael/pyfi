from pyfi.analytics.features.feature_engine import Processor
from pyfi.analytics.time_series.stats.inspect import Inspect
from pyfi.analytics.time_series.stats.correlation import Correlation
from pyfi.analytics.time_series.stats.cointegration import Cointegration
from pyfi.analytics.time_series.machine_learning.regression import RegressionPairs, RegType
from pyfi.analytics.time_series.stats.price_spread import PriceSpread
from pyfi.analytics.time_series.technical_analysis.ta import TechnicalAnalysis

import pandas as pd
import numpy as np
from enum import Enum
import pandas as pd

class Frequency(Enum):
    DAILY = 'D'
    WEEKLY_M = 'W-MON'
    WEEKLY_F = 'W-FRI'
    MONTHLY = 'M'

class AggFunc(Enum):
    MEAN = np.mean
    MEDIAN = np.median
    FIRST = 'first'
    LAST = 'last'


class TimeSeries(TechnicalAnalysis):

    def __init__( self,
                  df:pd.DataFrame,
                  dep_var:str = None,
                  indep_var:list = None,
                #   frequency:str = None,
                 ) -> None:
        """ Operates on a date/time indexed pandas DataFrame
        
        """
        super().__init__(df=df)

        self.df = df
        self.dep_var = dep_var
        self.indep_var = indep_var

        self.df.index.name = 'dt'

        if indep_var is not None:
            self.df = self.df[[dep_var, *indep_var]]

        self.prep = Processor(df=df, dep_var=dep_var)


    def group(self, frequency, aggfunc):
        """ Group by and aggregate

        """
        if frequency is None or aggfunc is None: raise ValueError ('Error - neither frequency nor aggfunc can be null')
        
        if not isinstance(aggfunc, str):
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).apply(aggfunc)
        elif aggfunc == AggFunc.LAST:
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).last()
        elif aggfunc == AggFunc.FIRST:
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).first()
        
        else: raise ValueError('Error - aggfunc parameter is not valid.')


    def curate(self):
        self.df = self.prep.curate()
        
    def winsorize(self, subset = None, limits = [.05, .05]):
        self.df = self.prep.winsorize_columns(df = self.df, subset = subset, limits=limits)

    def scale(self):
        self.df = self.prep.standard_scaler(df = self.df)
    
    def unscale(self):
        self.df = self.prep.inverse_standard_scaler(df = self.df)

    def check_stationarity(self, alpha = 0.05):
        return Inspect(df=self.df).check_stationarity(alpha=alpha)

    def decompose(self, var = None, period=7, plot=False):
        return Inspect(df=self.df).decompose(var=var, period=period, plot=plot)

    def correlate(self, plot = False, permutations = False):
        r2 = Correlation(df=self.df)
        
        self.pairs = r2.get_pairs()
        self.correlation = r2.get_correlation_tall()
        self.pearson_p_values = r2.get_pearson_p_values()

        if permutations:
            self.correlation_summary = r2.get_summary_as_permutations()
        else:
            self.correlation_summary = r2.get_correlation_summary()
        
        if plot:
            r2.plot_corr()
        
        return self.correlation_summary
        

    def cointegrate(self):
        ci = Cointegration(df = self.df)
        self.cointegratation = ci.get_cointegration_summary()
        self.cointegration_johansen = ci.get_cointegration_summary_johansen()

        return self.cointegratation, self.cointegration_johansen

    
    def regress(self, how=RegType.UNIVARIATE):
        """ Fit's a Regression model based on `Regression` object

        returns:
            summary: statsmodels.ols
        
        """
        reg = RegressionPairs(cls = self, how = how)
        
        return reg.get_summary()
    


    def get_price_spread(self, how = RegType.PERMUTATIONS, view=False):
        ps = PriceSpread(df = self.df, how=how)
        
        self.price_spread = ps.get_price_spread().reset_index().melt(id_vars='index', var_name = 'id').rename(columns={'index':'date'})

        self.price_spread_z_score = ps.get_price_spread_z_score().reset_index().melt(id_vars='index', var_name = 'id').rename(columns={'index':'date'})

        self.price_spread_adf = ps.get_price_spread_adf()

        return self.price_spread, self.price_spread_z_score, self.price_spread_adf
            
