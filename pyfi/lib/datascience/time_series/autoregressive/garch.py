from pyfi.lib.time_series.timeseries import TimeSeries

from arch import arch_model

import matplotlib.pyplot as plt
import pandas as pd

class Garch(TimeSeries):
    """  https://arch.readthedocs.io/en/latest/univariate/introduction.html
    """

    def __init__(   self, 
                    rets:pd.DataFrame=None, 
                    order:tuple = None,
                ):
        self.rets = rets
        self.order = order


    def fit(self): 
        """ 
        update_freq : int, optional
            Frequency of iteration updates.  Output is generated every
            `update_freq` iterations. Set to 0 to disable iterative output.
        disp : {bool, "off", "final"}
            Either 'final' to print optimization result or 'off' to display
            nothing. If using a boolean, False is "off" and True is "final"
        https://github.com/bashtage/arch/blob/main/arch/univariate/base.py
        """

        p,q = self.order

        model = arch_model(self.rets, vol='Garch', p=p, q=q)

        results = model.fit( update_freq=0, disp='off')
        
        return results


    def forecast(self, forecast_horizon=30,):
        results = self.fit()

        forecast = results.forecast(horizon=forecast_horizon)

        # print("Forecasted Volatility:")
        # print(forecast.mean.iloc[-1:]['h.01'].values[0])

        return forecast.mean.iloc[-1:]['h.01'].values[0] 
    

    def plot(self, results):
        
        # Summary of the model
        # print(results.summary())

        # Plot actual volatility and predicted volatility
        plt.figure(figsize=(10, 6))
        plt.plot(results.conditional_volatility, label='Predicted Volatility')
        plt.plot(self.rets, alpha=0.7, label='Actual Returns')
        plt.legend()
        plt.title('GARCH(1,1) - Actual vs Predicted Volatility')
        plt.xlabel('Date')
        plt.ylabel('Returns/Volatility')
        plt.show()

        # Residual analysis
        residuals = results.resid
        fig = plt.figure(figsize=(10, 6))
        plt.title('Residuals of GARCH(1,1) Model')
        plt.plot(residuals)
        plt.xlabel('Date')
        plt.ylabel('Residuals')
        plt.show()

        # base functionality
        fig = results.plot(annualize="D")
        plt.show()

