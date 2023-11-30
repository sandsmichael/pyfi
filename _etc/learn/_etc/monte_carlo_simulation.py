""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                                                                                                                │
  │ # A monte carlo simulation on daily equity returns used to predict future price level
                           │
  │ # https://www.interviewqs.com/blog/intro-monte-carlo                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import seaborn as sns
import pandas_market_calendars as mcal
import datetime
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings('ignore')

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Helper Function                                                                                                  │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Main                                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class NormalDistribution:

    def __init__(self, data:pd.DataFrame(), col:str = None) -> None:
        
        self.data = data
        self.col = col
        self.number_of_trading_days = 252 

        self.get_periods()
        self.get_growth()
        self.get_std_dev()
    
    def __str__(self):
        return self.serialize()

    def get_periods(self):
        self.n_days = (self.data.index[-1] - self.data.index[0]).days
        self.n_years = self.n_days / 365.0

    def get_growth(self):
        total_growth = (self.data[self.col][-1] / self.data[self.col][1])
        self.cagr = total_growth ** (1/self.n_years) - 1 # mean annual growth rate; raise the total growth to the inverse of the # of years (e.g. ~1/10) to annualize growth rate

    def get_std_dev(self):
        self.std_dev = self.data[self.col].pct_change().std()         # calculate the standard deviation of the daily price changes
        self.std_dev = self.std_dev * math.sqrt(self.number_of_trading_days) # scale std_dev by an annualization factor reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx

    def serialize(self):
        return 'NormalDistributionObject: ' + str({
            'cagr':self.cagr,
            'std_dev':self.std_dev,
            'days':self.n_days,
            'years':self.n_years
        })


    @staticmethod
    def periodic_distribution_chart(ax):
        data1 =  yf.download(TICKER, start="2020-01-01", end="2020-12-31", progress = False)
        data2 =  yf.download(TICKER, start="2021-01-01", end="2021-12-31", progress = False)
        data3 =  yf.download(TICKER, start="2022-01-01", end="2022-12-31", progress = False)
        x = data1['Adj Close'].to_frame().pct_change().rename(columns = {'Adj Close':'2020'}).merge(
            data2['Adj Close'].to_frame().pct_change().rename(columns = {'Adj Close':'2021'}), how = 'outer', left_index = True, right_index = True
        ).merge(
            data3['Adj Close'].to_frame().pct_change().rename(columns = {'Adj Close':'2022'}), how = 'outer', left_index = True, right_index = True
        )
        for c in x.columns:
            sns.distplot(x[c], hist = False, kde = True,  kde_kws = {'linewidth': 3}, label = c, ax = ax)
        # Plot formatting
        ax.legend( title = 'Year') # prop={'size': 13},
        plt.title('Density Plot of Annual Returns')
        ax.set_xlabel('Percent Change')
        ax.set_ylabel('Density')
        return ax


    @staticmethod
    def normal_distribution_pdf_and_hist_chart(ax1):
        mu, sigma = dist.cagr/dist.number_of_trading_days, dist.std_dev/math.sqrt(dist.number_of_trading_days) # unannualize cagr to get average daily percent change; unnanualize the std dev that was previously annualized above
        s = np.random.normal(mu, sigma, len(dist.data[dist.col]))
        counts, bins, ignored = ax1.hist(dist.data[dist.col].pct_change(), density = True, bins = 40, label = 'Normalized bins') # norm=True thus returns a histogram which can be interpreted as a probability distribution.
        ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),  linewidth=3, color='y', label = 'PDF')
        ax1.set_title(f'Distribution of Returns') 
        ax1.set_xlabel('Daily Percent Change')
        ax1.set_ylabel('Density')
        ax1.axvline(0, c='black', label = 'Zero')
        ax1.axvline(dist.cagr/dist.number_of_trading_days, c='lightgreen', label = 'Mean')
        ax1.annotate(f'Mean: {round(mu,4) *100}%', xy=(0.03, 0.85), xycoords=('data', 'axes fraction'),xytext=(0, 18), textcoords='offset points', va='top', ha='center')
        ax1.annotate(f'Std Dev: {round(dist.std_dev,4)}', xy=(0.03, 0.80), xycoords=('data', 'axes fraction'),xytext=(0, 18), textcoords='offset points', va='top', ha='center')
        ax1.legend()



class MonteCarlo:

    def __init__(self) -> None:
        pass

    @classmethod
    def random_walk(self, dist:NormalDistribution, number_of_trials=100):
        '''
        Generate random values for 1 year's worth of trading (252 days), using numpy and assuming a normal distribution with mean of dialy growth rate and standard dev of daily standard deviation of percent change in price
        Now that we have created a random series of future daily return %s, we can simply apply these forward-looking to our last stock price in the window, effectively carrying forward  a price prediction for the next year
        '''
        closing_prices = []
        simulation_prices = []
        for i in range(number_of_trials):

            daily_return_percentages = np.random.normal(dist.cagr/dist.number_of_trading_days, dist.std_dev/math.sqrt(dist.number_of_trading_days), dist.number_of_trading_days) + 1 
            
            price_series = [dist.data[dist.col][-1]] # the last actual value provided in our input data
            
            for j in daily_return_percentages:
                price_series.append(price_series[-1] * j)
            
            simulation_prices.append(price_series)
        
            closing_prices.append(price_series[-1])

        self.closing_prices = closing_prices
        self.simulation_prices =  pd.DataFrame(simulation_prices).transpose()
        self.mean_end_price = round(np.mean(self.closing_prices),2)

    
    def simulation_time_series(self, ax2):
        nyse = mcal.get_calendar('NYSE')
        # get historical prices (actuals)
        history = dist.data[[dist.col]].reset_index()
        history['Date'] = pd.to_datetime(history['Date']).dt.date
        history.set_index('Date', inplace = True)
        # construct a range of future dates to set as index for the simulated prices
        future = self.simulation_prices
        date_range = nyse.valid_days(start_date=dist.data.reset_index()['Date'].iloc[-1], end_date= '2030-01-01')[1:len(self.simulation_prices)+1]
        future['Date'] = date_range
        future['Date'] = future['Date'].dt.date
        future.set_index('Date', inplace = True)
        self.simulation_prices = history.merge(future, how = 'outer', left_index = True, right_index = True)
        ax2.plot(self.simulation_prices)
        ax2.set_title('Monte Carlo Simulation')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity Price')
        ax2.annotate(f'Last Actual: {round(history.iloc[-1].values[0],4)}', xy=(mdates.date2num(pd.to_datetime('2022-01-01')), history.iloc[-1].values[0]+(0.4 * history.iloc[-1].values[0])))
        ax2.annotate(f'Mean Ending: {round(self.mean_end_price,4)}', xy=(mdates.date2num(pd.to_datetime('2022-01-01')), history.iloc[-1].values[0]+(0.2 * history.iloc[-1].values[0])))


    def simulation_results_histogram(self, ax):
        top_ten = np.percentile(self.closing_prices,100-10)
        bottom_ten = np.percentile(self.closing_prices,10)
        counts, bins, ignored = ax.hist(self.closing_prices,bins=40, label = 'Frequency bins')
        ax.axvline(top_ten,color='r',linestyle='dashed',linewidth=2, label ='Top 10%')
        ax.axvline(bottom_ten,color='r',linestyle='dashed',linewidth=2, label ='Bottom 10%')
        ax.axvline(dist.data[dist.col][-1],color='lightgreen', linestyle='dashed',linewidth=2, label ='Actual')
        ax.axvline(self.mean_end_price,color='black', linewidth=2, label ='Mean')
        ax.set_title('Simulation Results - Ending Prices')
        ax.set_xlabel('Equity Price')
        ax.set_ylabel('Frequency')
        ax.legend(loc = 'best')
        # Label the raw counts and the percentages below the x-axis
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts, bin_centers):
            # Label the raw counts
            ax.annotate('{:.0f}'.format(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, 18), textcoords='offset points', va='top', ha='center')


    def plot(self, dist:NormalDistribution):
        # fig, axs = plt.subplots(3,2, figsize=(15, 10))
        
        fig = plt.figure(figsize=(15, 10), constrained_layout=True) # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html
        gs = GridSpec(3, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, : -1])        
        ax2 = fig.add_subplot(gs[1, :-1])
        ax3 = fig.add_subplot(gs[2, :-1])
        ax5 = fig.add_subplot(gs[2:, -1])
        ax4 = fig.add_subplot(gs[:2, -1])
        fig.suptitle(f'{TICKER} Returns - From {START_DATE} To {END_DATE}')

        dist.normal_distribution_pdf_and_hist_chart(ax1)
        self.simulation_time_series(ax2)
        self.simulation_results_histogram(ax3)
        dist.periodic_distribution_chart(ax4)

        results_description = self.simulation_prices.drop('Adj Close', axis=1).dropna(how = 'any', axis=0).iloc[-1].describe().round(2).reset_index() 
        fig, ax5 = render_mpl_table(results_description, header_columns=0, col_width=2.0, ax=ax5)
        ax5.set_title('Summary of Simulation Results')
        # plt.tight_layout()
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.savefig('./_img/monte_carlo_figure.png', bbox_inches = 'tight')
        plt.show()

        print(self.simulation_prices)



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Run                                                                                                              │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
TICKER = 'SPY'
START_DATE = "2019-12-25"
END_DATE = "2022-12-25"
data =  yf.download(TICKER, start=START_DATE, end=END_DATE, progress = False)
dist = NormalDistribution(data = data, col = 'Adj Close')
print(dist)

mc = MonteCarlo()
mc.random_walk(dist = dist)
mc.plot(dist = dist)

