from pyfi.base.preprocessors.prep import Prep
from pyfi.analytics.time_series.stats.inspect import Inspect
from pyfi.analytics.time_series.stats.correlation import Correlation

import pandas as pd
import numpy as np
from enum import Enum

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

class TimeSeries:

    def __init__( self,
                  df:pd.DataFrame,
                  dep_var:str = None,
                  indep_var:list = None,
                  frequency:str = None,
                 ) -> None:
        """ Operates on a date/time indexed pandas DataFrame
        
        """
        self.df = df
        self.dep_var = dep_var
        self.indep_var = indep_var

        self.df.index.name = 'dt'

        if indep_var is not None:
            self.df = self.df[[dep_var, *indep_var]]

        self.prep = Prep()


    def group(self, frequency, aggfunc):
        """ Group by and aggregate

        """
        if frequency is None: raise ValueError ('Error - frequency can\'t be null')

        if aggfunc is None: raise ValueError ('Error - aggfunc can\'t be null')
        
        if not isinstance(aggfunc.value, str):
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).apply(aggfunc)
        elif aggfunc == AggFunc.LAST:
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).last()
        elif aggfunc == AggFunc.FIRST:
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).first()
        
        else: raise ValueError('Error - aggfunc parameter is not valid.')


    def curate(self):
        self.df = self.prep.curate(df = self.df)
        
    def winsorize(self, subset = None, limits = [.05, .05]):
        self.df = self.prep.winsorize_columns(df = self.df, subset = subset, limits=limits)

    def scale(self):
        self.df = self.prep.standard_scaler(df = self.df)
    
    def unscale(self):
        self.df = self.prep.inverse_standard_scaler(df = self.df)

    def check_stationarity(self, alpha = 0.05):
        self.adf = Inspect(df=self.df).check_stationarity(alpha=alpha)

    def decompose(self, var = None, period=7, plot=False):
        self.decomposed = Inspect(df=self.df).decompose(var=var, period=period, plot=plot)

    def correlate(self, plot = False):
        r2 = Correlation(df=self.df)
        self.pairs = r2.get_pairs()
        self.correlation = r2.get_correlation_tall()
        self.pearson_p_values = r2.get_pearson_p_values()
        self.correlation_summary = r2.get_correlation_summary()
        if plot:
            r2.plot_corr()
        
