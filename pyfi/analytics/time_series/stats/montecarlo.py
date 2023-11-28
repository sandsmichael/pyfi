import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MonteCarlo:


    def __init__(self):
        pass


    def run(self):

        initial_price = 100  # Initial stock price
        drift = 0.05  # Average annual return
        volatility = 0.2  # Volatility (standard deviation of returns)
        days = 252  
        num_simulations = 10  

        daily_returns = np.random.normal((drift / days), volatility / np.sqrt(days), (days, num_simulations))

        price_series = initial_price * (1 + daily_returns).cumprod(axis=0)

        df = pd.DataFrame(price_series)

        # Plot simulations
        # plt.figure(figsize=(10, 6))
        # plt.plot(df)
        # plt.title('Monte Carlo Simulation of Stock Prices')
        # plt.xlabel('Days')
        # plt.ylabel('Stock Price')
        # plt.legend([f'Simulation {i+1}' for i in range(num_simulations)])
        # plt.show()
