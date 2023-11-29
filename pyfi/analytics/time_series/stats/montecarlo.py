import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_market_calendars import get_calendar
from pyfi.analytics.time_series.stats.descriptive import Descriptive


class MonteCarlo:

    def __init__(self, prices:pd.Series = None, num_simulations:int = 10, n_periods:int = 252):
        
        self.prices = prices

        self.prices.index = [d.date() for d in self.prices.index]

        self.log_rets = np.log(1 + self.prices.pct_change())

        self.num_simulations = num_simulations

        self.n_periods = n_periods   


    def run(self):
        # params
        num_simulations = self.num_simulations  

        n_periods = self.n_periods

        u = self.log_rets.mean()

        var = self.log_rets.var()
        
        drift = u - (0.5 * var)
        
        days = (self.log_rets.index[-1] - self.log_rets.index[0]).days
        
        cagr = ((((self.log_rets[-1]) / self.log_rets[0])) ** (365.0/days)) - 1
        
        vol = self.log_rets.std() 

        # random walk based on normal distribution
        daily_returns = np.random.normal( (drift / days), vol, (n_periods, num_simulations) )

        simulated = pd.DataFrame( self.prices[-1] * (1 + daily_returns).cumprod(axis=0) )

        # format results
        simulated.columns = [str(n) for n in range(num_simulations)]

        reshaped_df = pd.DataFrame({f'{i}': self.prices for i in range(num_simulations)})

        # stack simulation results ontop of historical observations
        most_recent_date = self.prices.index[-1]

        simulated.index = self.get_future_business_days(start_date = most_recent_date, n = n_periods)

        full_simulation = pd.concat([reshaped_df, simulated], axis=0)

        self.simulated = simulated

        self.full_simulation = full_simulation

        return simulated, full_simulation


    def describe(self):
        return Descriptive(df=self.simulated.iloc[-1].T.to_frame(name='Simulation End Values')).describe()


    def get_future_business_days(self, start_date, n):
        nyse = get_calendar('XNYS')
        dates_list = nyse.valid_days(start_date=start_date, 
                                     end_date=pd.to_datetime(start_date) + pd.offsets.BDay(n+100))
        dates_list = [pd.to_datetime(d).date() for d in dates_list[:n]]
        return dates_list
    

    def plot(self):

        # line
        plt.figure(figsize=(10, 6))
        plt.plot(self.full_simulation)
        plt.title('Monte Carlo Simulation of Stock Prices')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        # plt.legend([f'Simulation {i+1}' for i in range(num_simulations)])
        plt.show()

        # hist
        plt.figure(figsize=(10, 6))
        plt.hist(self.simulated.iloc[-1].T)
        plt.title('Simulation Ending Prices')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        # plt.legend([f'Simulation {i+1}' for i in range(num_simulations)])
        plt.show()
