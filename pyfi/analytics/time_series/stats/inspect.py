import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


class Inspect:

    def __init__(self, df) -> None:
        self.df = df


    def check_stationarity(self, alpha = 0.05):
        """ 
        alpha: significance level
        bool: True if series is stationary, False if it is not stationary
        
        ADF H0 --> Series has a unit root and is therefore not stationary. 
        A unit root indicates that the series has some dependence between consecutive observations, 
        suggesting that it doesn't exhibit constant mean and variance over time.
        """
        result = {}
        for col in self.df.columns:
            adf_result = adfuller(self.df[col])
            
            result[col] = {'ADF Statistic': adf_result[0], 'p-value': adf_result[1], 'lags': adf_result[2], 'n-obs': adf_result[3]}
            for key, value in adf_result[4].items():
                result[col][f'Critical Value {key}'] = value
        
        res = pd.DataFrame(result).transpose()
        res['bool'] = [True if x <= alpha else False for x in res['p-value']]
        return res


    def decompose(self, var = None, period = 7, plot=False):

        if var is None: raise ValueError('var parameter can not be None. Specify a column to decompose')

        data = self.df[var]

        decomposition = seasonal_decompose(data, model='additive', period=period)  # 'additive' model for an additive decomposition
        
        valid_rows = ~np.isnan(decomposition.trend) & ~np.isnan(decomposition.resid)
        valid_index = decomposition.trend.index[valid_rows]

        df_decomposed = pd.DataFrame({
            'Trend': decomposition.trend[valid_index],
            'Seasonal': decomposition.seasonal[valid_index],
            'Residuals': decomposition.resid[valid_index]
        })

        if plot:
            plt.figure(figsize=(10, 8))

            plt.subplot(411)
            plt.plot(data, label='Original Time Series')
            plt.legend()

            plt.subplot(412)
            plt.plot(decomposition.trend, label='Trend')
            plt.legend()

            plt.subplot(413)
            plt.plot(decomposition.seasonal, label='Seasonal')
            plt.legend()

            plt.subplot(414)
            plt.plot(decomposition.resid, label='Residuals')
            plt.legend()

            plt.tight_layout()
            plt.show()
        
        return df_decomposed


