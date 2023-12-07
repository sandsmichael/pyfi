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
        return "Add logic to process series here..."


    def describe(self):

        if isinstance(self.df, pd.DataFrame):
            return self.process_df()
        
        else:
            return self.process_series()



