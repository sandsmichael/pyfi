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







# def simulate_random_portfolios(num_iterations, mean_returns, df, cov, rf, tickers):
#     results_matrix = np.zeros((len(mean_returns) + 2, num_iterations))  # columns, by rows
#     for i in range(num_iterations):
#         print('Calculating Portfolio: #' + str(i))
#         weights = []
#         for i in range(num_iterations):
#             # Select random weights and normalize to set the sum to 1
#             weights = np.array(np.random.random(ntickers))
#             weights /= np.sum(weights)

#         weights = np.random.random(len(mean_returns))
#         weights /= np.sum(weights)

#         # sharpe_ratio, sortino_ratio = calc_portfolio_perf(weights, mean_returns, df, cov, rf)
#         results_matrix[0, i] = sharpe_ratio
#         results_matrix[1, i] = sortino_ratio

#         # iterate through the weight vector and add data to results array
#         for j in range(len(weights)):
#             results_matrix[j + 2, i] = weights[j]

#     results_df = pd.DataFrame(results_matrix.T, columns=['sharpe', 'sortino'] + [ticker for ticker in tickers])
#     return results_df

# results_frame = simulate_random_portfolios(num_iterations, mean_returns, df, cov, rf, tickers)
# print(results_frame.head())
# def run_montecarlo(ticker, historic_path):
   
#     df = pd.read_table(historic_path, sep=',', skiprows=range(0, 2), names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj.Close', 'Volume'])
#     df_indexed = df.set_index('Date')
     
#     start_day = df_indexed.index[0]
#     end_day = df_indexed.index[-1]
#     date_range = 252   
#     # COMPOUND ANNUAL GROWTH RATE
#     cagr = ((((df_indexed['Adj.Close'][-1]) / df_indexed['Adj.Close'][1])) ** (365.0 / date_range)) - 1
#     # print ('CAGR =', str(round(cagr, 4) * 100) + "%")
#     mu = cagr
#     # create a series of percentage returns and calculate the annual volatility of returns
#     df_indexed['Returns'] = df_indexed['Adj.Close'].pct_change()
#     volatility = df_indexed['Returns'].std() * np.sqrt(252)
#     # print ("Annual Volatility =", str(round(volatility, 4) * 100) + "%")
#     # MONTE CARLO VARIABLES
#     S = df_indexed['Adj.Close'][-1]  # starting stock price (i.e. last available real stock price)
#     T = 252  # Number of trading days
#     mu = cagr  # Return
#     vol = volatility  # Volatility
#     # create list of daily returns using random normal distribution
#     daily_returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
#     # set starting price and create price series generated by above random daily returns
#     price_list = [S]
#     result = []
#     plt.clf()
#     for i in range(1000): # number of simulations to run
#         # create list of daily returns using random normal distribution
#         daily_returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
#         # set starting price and create price series generated by above random daily returns
#         price_list = [S]
#         for x in daily_returns:
#             price_list.append(price_list[-1] * x)

#         # plot data from each individual run which we will plot at the end
#         # plt.plot(price_list)
#         result.append(price_list[-1])
#     mean = round(np.mean(result), 2)
#     print(mean)
#     percent5 = np.percentile(result, 5)
#     percent95 = np.percentile(result, 95)

#     outlist = [mean, percent5, percent95, volatility, cagr]
#     # print(simulation)


#     return outlist




# # plot 
# def plot():
#     path_sim = path_output + 'MonteCarlo_Simulation.png'
#     plt.savefig(path_sim, bbox_inches='tight')

#     plt.clf()
#     #plt.show()
#     plt.hist(result, bins=50)
#     path_hist = path_output + 'MonteCarlo_Histogram.png'
#     plt.savefig(path_hist, bbox_inches='tight')
#     plt.clf()
#     #plt.show() 

#     plt.hist(result, bins=100)
#     plt.axvline(np.percentile(result, 5), color='r', linestyle='dashed', linewidth=2)
#     plt.axvline(np.percentile(result, 95), color='r', linestyle='dashed', linewidth=2)
#     patquanthist = path_output + 'MonteCarlo_QuantHisto.png'
#     print(patquanthist)
#     plt.savefig(patquanthist, bbox_inches='tight')
#     plt.clf()
#     plt.show()
         





# #locate position of portfolio with highest Sharpe Ratio
# max_sharpe_port = simulation_results_frame.iloc[simulation_results_frame['sharpe'].idxmax()]
# #locate positon of portfolio with minimum standard deviation
# min_vol_port = simulation_results_frame.iloc[simulation_results_frame['stdev'].idxmin()]
# #create scatter plot coloured by Sharpe Ratio
# plt.subplots(figsize=(15,10))
# plt.scatter(simulation_results_frame.stdev,simulation_results_frame.ret,c=simulation_results_frame.sharpe,cmap='RdYlBu')
# plt.xlabel('Standard Deviation')
# plt.ylabel('Returns')
# plt.colorbar()
# #plot red star to highlight position of portfolio with highest Sharpe Ratio
# plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)
# #plot green star to highlight position of minimum variance portfolio
# plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
# plt.show()

# max_sharpe = max_sharpe_port.to_frame().T
# min_vol = min_vol_port.to_frame().T

# poutdir3 = cwd + poutdir + 'Max_sharpe.xlsx'
# poutdir2 = cwd + poutdir + 'Min_Vol.xlsx'
# max_sharpe.to_excel(poutdir3)
# min_vol.to_excel(poutdir2)


# # print(max_sharpe)
# # print(min_vol)
