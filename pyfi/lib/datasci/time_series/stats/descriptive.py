import pandas as pd
import numpy as np

class Descriptive():

    def __init__(self, df) -> None:

        self.df = df

    def process_df(self):
        # count = self.df.size().to_frame(name = 'size').T
        quantile = self.df.quantile([.1, .25, .5, .75,  .9])
        mean = self.df.mean().to_frame(name = 'mean').T
        median = self.df.median().to_frame(name = 'median').T
        std = self.df.std().to_frame(name = 'std').T
        var = self.df.var().to_frame(name = 'var').T
        skew = self.df.skew().to_frame(name = 'skew').T
        kurtosis = self.df.kurtosis().to_frame(name = 'kurtosis').T
        excess_kurtosis = (kurtosis - 3).rename(index={'kurtosis':'excess kurtosis'})
        cv = self.df.std() / self.df.mean().to_frame(name = 'cv').T
        min = self.df.min().to_frame(name = 'min').T
        max = self.df.max().to_frame(name = 'max').T
        return pd.concat([quantile, mean, median, std, var, skew, kurtosis, excess_kurtosis, cv, min, max])


    def process_series(self):
        quantile = self.df.quantile([.1, .25, .5, .75, .9]).to_frame()
        mean = pd.Series(self.df.mean(), name='mean').to_frame().T
        median = pd.Series(self.df.median(), name='median').to_frame().T
        std = pd.Series(self.df.std(), name='std').to_frame().T
        var = pd.Series(self.df.var(), name='var').to_frame().T
        skew = pd.Series(self.df.skew(), name='skew').to_frame().T
        kurtosis = pd.Series(self.df.kurtosis(), name='kurtosis').to_frame().T
        excess_kurtosis = pd.Series(kurtosis.values[0] - 3, name='excess kurtosis').to_frame().T
        cv = (pd.Series(self.df.std() / self.df.mean(), name='cv')).to_frame().T
        min_value = pd.Series(self.df.min(), name='min').to_frame().T
        max_value = pd.Series(self.df.max(), name='max').to_frame().T

        result = pd.concat([quantile, mean, median, std, var, skew, kurtosis, excess_kurtosis, cv, min_value, max_value])
        result.index = result.index.rename(None)
        return result


    def describe(self):

        if isinstance(self.df, pd.DataFrame):
            return self.process_df()
        
        else:
            return self.process_series()



