import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

class Prep:

    def __init__(self) -> None:
        pass

    def curate(self, df):
        return df.dropna(how = 'any', axis=0) # Dropna or fill with mean?

    def winsorize_columns(self, df, subset = None, limits=[0.05, 0.05]):
        winsorized_df = df.copy()

        if subset is None:
            subset = winsorized_df.columns

        for column in subset:
            winsorized_df[column] = winsorize(df[column], limits=limits)

        return winsorized_df
    

    def standard_scaler(self, df):
        """ Standardize each series of a dataframe by (observation - mean) /  standard deviation 
        """
        self.scaler = StandardScaler()
        self.scaled_values = self.scaler.fit_transform(df)
        return pd.DataFrame(self.scaled_values, columns=df.columns)
        

    def inverse_standard_scaler(self, df):
        unscaled_values = self.scaler.inverse_transform(df)
        return pd.DataFrame(unscaled_values, columns=df.columns)
