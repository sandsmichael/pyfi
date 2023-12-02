from pyfi.base.retrievers import equity
from pyfi.core.timeseries import TimeSeries

from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

import matplotlib.pyplot as plt
import pandas as pd

class Garch(TimeSeries):

    def __init__(   self, 
                    rets:pd.DataFrame=None, 
                    order:tuple = None,
                ):
        self.rets = rets
        self.order = order


    def run(self, forecast_horizon):    
        p,q = self.order

        model = arch_model(self.rets, vol='Garch', p=p, q=q)

        # Fit the model
        results = model.fit()

        # Summary of the model
        print(results.summary())

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


        # Diagnostic tests (Ljung-Box test for autocorrelation in residuals)
        # lb_test = acorr_ljungbox(residuals, lags=[10])
        # print("Ljung-Box test p-value:", lb_test[1])

        # Out-of-sample forecasting evaluation
        forecast = results.forecast(horizon=forecast_horizon)

        # Print forecasted volatility
        print("Forecasted Volatility:")
        print(forecast.mean.iloc[-1:])

        # Actual volatility for the forecast horizon (for comparison)
        actual_volatility = self.rets[-forecast_horizon:]
        print("Actual Volatility:")
        print(actual_volatility)