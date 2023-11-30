# coding=utf-8

import math
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
from statsmodels import regression
import pandas as pd
import numpy as np
import pandas_datareader as web
import statsmodels.formula.api as smf
import datetime
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Helper Function                                                                                                  │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=8,
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


def get_month_end(start_date = datetime.datetime.today(), offset:int = None):
    first = start_date.replace(day=1)
    last_month_end = first - datetime.timedelta(days=1)
    if offset == None:
        return last_month_end
    else:
        return last_month_end + relativedelta(months=offset)
        

class Equity:

    def __init__(self,  VENDOR = 'Yahoo', SECURITY = None, BENCHMARK = None, START_DATE= None, END_DATE = None, RETURN_COL = None) -> None:
        
        self.SECURITY = SECURITY
        self.BENCHMARK = BENCHMARK
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE
        self.RETURN_COL = RETURN_COL
        self.VENDOR = VENDOR

        if self.VENDOR == 'Yahoo':
            self.get_prices()
        
        elif self.VENDOR == 'Morningstar':
            self.get_morningstar_prices()


    def get_prices(self):
        self.prices =  yf.download([self.SECURITY, self.BENCHMARK], start=self.START_DATE, end=self.END_DATE, progress = False)
        return self.prices

    def get_morningstar_prices(self):
        df = pd.read_excel("C:\data\Daily Return Index.xlsx").rename(columns = {'Unnamed: 0':'Date'}).set_index('Date')
        df.columns = pd.MultiIndex.from_product([df.columns, [self.RETURN_COL]])
        df = df.swaplevel(0, 1, axis = 1)
        df = df.iloc[:, df.columns.get_level_values(1).isin([self.SECURITY, self.BENCHMARK])]

        self.prices = df
        return self.prices

    def get_returns(self):
        if self.VENDOR == 'Yahoo':
            self.returns =  self.get_prices()[self.RETURN_COL].pct_change().dropna(how = 'all', axis=0)
        elif self.VENDOR == 'Morningstar':
            self.returns =  self.get_morningstar_prices()[self.RETURN_COL].pct_change().dropna(how = 'all', axis=0)
        return self.returns


    def get_cumulative_returns(self):
        self.cum_ret = self.get_returns().cumsum()        
        return self.cum_ret


    def get_volitility(self): # FIXME
        daily = self.get_returns().std()
        monthly = math.sqrt(21) * daily # Assume 21 trading days in a month
        annual = math.sqrt(252) * daily # Assume 252 trading days in a year
        return daily, monthly, annual
    

    def get_rolling_volitlity(self, window = 5):
        return  self.get_returns().rolling(window).std()


    def get_cumulative_volitlity(self, window = 5):
        return self.get_rolling_volitlity(window = window).cumsum()


    def get_beta(self):
        returns = self.get_returns()

        X = returns[self.BENCHMARK].values
        Y = returns[self.SECURITY].values

        def linreg(x,y):
            x = sm.add_constant(x)
            model = regression.linear_model.OLS(y,x).fit()
            x = x[:, 1]
            return model.params[0], model.params[1]

        alpha, beta = linreg(X,Y)
        return alpha, beta, X, Y


    def get_rolling_beta(self, window=5):
        returns = self.get_returns()
        return (returns.rolling(window).cov().unstack()[self.BENCHMARK][self.SECURITY] / returns[self.BENCHMARK].rolling(window).var()).to_frame().reset_index().rename(columns = {0:'Rolling Beta'})


    def get_sharpe_ratio(self, y, window, risk_free_rate):
        mean_daily_return = sum(y) / len(y)
        s = y.std()
        daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
        sharpe_ratio = 252**(window/252) * daily_sharpe_ratio     # annualized   
        return sharpe_ratio


    def get_rolling_sharpe_ratio(self, window, risk_free_rate):
        returns = self.get_returns()
        return returns.rolling(window).apply( lambda y : self.get_sharpe_ratio(y, window = window, risk_free_rate =risk_free_rate ))


    def get_drawdowns(self):
        prices = self.get_prices()[self.RETURN_COL].reset_index()
        xs = prices[self.SECURITY]
        i = np.argsort(np.maximum.accumulate(xs) - xs).iloc[-1]
        print(prices['Date'].iloc[i])
        j = np.argsort(xs[:i]).iloc[-1] 
        print(prices['Date'].iloc[j])
        plt.plot(xs)
        plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
        plt.show()


    def get_famma_french_attribution(self):

        returns = self.get_returns().reset_index()
        returns['Date'] = pd.to_datetime(returns['Date'], format = "%Y-%m-%d")
        returns.rename(columns = {self.SECURITY: 'portf_rtn'}, inplace = True)
        returns.set_index('Date', inplace = True)
        grouped_returns = returns.groupby(pd.Grouper(freq='M')).sum()['portf_rtn'].to_frame()

    
        def rolling_factor_model(input_data, formula, window_size):
            '''
            Function for estimating the Fama-French (n-factor) model using a rolling window of fixed size.
            Parameters
            ------------
            input_data : pd.DataFrame
                A DataFrame containing the factors and asset/portfolio returns
            formula : str
                `statsmodels` compatible formula representing the OLS regression  
            window_size : int
                Rolling window length.
            Returns
            -----------
            coeffs_df : pd.DataFrame
                DataFrame containing the intercept and the three factors for each iteration.
            '''
            coeffs = []

            for start_index in range(len(input_data) - window_size + 1):
                end_index = start_index + window_size

                # define and fit the regression model
                ff_model = smf.ols(
                    formula=formula,
                    data=input_data[start_index:end_index]
                ).fit()

                # store coefficients
                coeffs.append(ff_model.params)

            coeffs_df = pd.DataFrame(
                coeffs,
                index=input_data.index[window_size - 1:]
            )

            return coeffs_df

        df = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=self.START_DATE)[0]
        
        df = df.div(100)
        
        df.index = df.index.strftime('%Y-%m-%d')
        
        df.index.name = 'Date'
        
        df.index = pd.to_datetime(df.index)

        ff_data = df.merge(grouped_returns, left_index=True, right_index = True, how = 'inner')

        ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
        
        ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf

        ff_model = smf.ols(formula='portf_ex_rtn ~ mkt + smb + hml', data=ff_data).fit()
        
        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])

        MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
        
        results_df = rolling_factor_model(ff_data,MODEL_FORMULA, window_size=3)
        
        return ff_model.summary(), results_df


    def get_trailing_returns(self):
        print(self.prices)
        # market_days = nyse.valid_days(start_date = '1900-01-01', end_date = '2100-01-01')
        # prev_month_end_1 = get_month_end(offset = -1)
        # prev_month_end_2 = print(get_month_end(offset = -2))
        # prev_month_end_3 = print(get_month_end(offset = -3))
        def get(df):
            one_day = ((df.iloc[-1] / df.iloc[-2]) - 1).values[0]
            five_day = ((df.iloc[-1] / df.iloc[-5]) - 1).values[0]
            twenty_day =( (df.iloc[-1] / df.iloc[-20]) - 1).values[0]
            sixty_day = ((df.iloc[-1] / df.iloc[-60]) - 1).values[0]
            one_twenty_day = ((df.iloc[-1] / df.iloc[-120]) - 1).values[0]
            two_fifty_two_day = ((df.iloc[-1] / df.iloc[-252]) - 1).values[0]
            return [one_day, five_day, twenty_day, sixty_day, one_twenty_day, two_fifty_two_day]

        security = get(self.prices.iloc[:, self.prices.columns.get_level_values(1) == self.SECURITY])
        benchmark = get(self.prices.iloc[:, self.prices.columns.get_level_values(1) == self.BENCHMARK])
        df = pd.DataFrame([security, benchmark], columns = ['1D','5D', '20D', '60D', '120D', '252D'])
        excess = (df.iloc[0] - df.iloc[1]).to_frame().T
        df = pd.concat([df, excess], axis =0)
        df = df.multiply(100).round(2)
        df['Name'] = ['Inv.', 'BM', 'Excess']
        df = df[['Name'] + list([x for x in df.columns if x != 'Name'])]
        return df




class TearSheet:

    def __init__(self) -> None:
        pass

    @classmethod
    def cumulative_returns_chart(self, eqobj:Equity, ax = None):
        df = eqobj.get_cumulative_returns()
        df.plot(ax = ax, grid = True)
        ax.set_title('Cumulative Returns')
        ax.axhline(0, color = 'black')
        ax.axhline(0, color = 'black')
        ax.legend(ncol = 2, prop={'size':6})


    @classmethod
    def daily_returns_chart(self, eqobj:Equity, ax = None):
        df = eqobj.get_returns()
        df.plot(ax = ax, grid = True)
        ax.set_title('Daily Returns')
        ax.axhline(0, color = 'black')
        ax.legend(ncol = 2, prop={'size':6})


    @classmethod
    def daily_return_distribution(self, eqobj:Equity, ax=None):
        df = eqobj.get_returns()
        # df.plot(ax = ax, grid = True, kind='hist', stacked=True)
        melt = df.reset_index().melt(id_vars = ['Date'], value_name='Return', var_name = 'Security')
        sns.histplot(data=melt, x="Return", hue="Security", multiple="stack", ax = ax)
        ax.set_title('Daily Returns')
        ax.axhline(0, color = 'black')
        ax.legend(ncol = 2, prop={'size':6})


    @classmethod
    def rolling_std_dev_chart(self, eqobj:Equity, ax = None):
        rolling_std = eqobj.get_rolling_volitlity(window = 20)
        rolling_std.plot(ax = ax, grid = True)
        ax.set_title('Rolling Volitlity')
        ax.legend(ncol = 2, prop={'size':6})


    @classmethod
    def cumulative_std_dev_chart(self, eqobj:Equity):
        rolling_std = eqobj.get_cumulative_volitlity(window = 20).reset_index()
        melt = rolling_std.melt(id_vars = ['Date'], var_name = 'Ticker', value_name = 'Cumulative Std Dev')
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.lineplot(data = melt, x = 'Date', y = 'Cumulative Std Dev', hue = 'Ticker', ax = ax)
        ax.set_title('Cumulative Standard Deviation')


    @classmethod
    def beta_chart(self, eqobj, ax = None):
        alpha, beta, X, Y = eqobj.get_beta()
        X2 = np.linspace(X.min(), X.max(), 100)
        Y_hat = X2 * beta + alpha
        ax.scatter(X, Y, alpha=0.3) # Plot the raw data
        ax.plot(X2, Y_hat, 'r', alpha=0.9)
        print(alpha, beta)
        ax.set_title(f'Beta of {eqobj.SECURITY} and {eqobj.BENCHMARK}')
        ax.set_xlabel(eqobj.BENCHMARK)
        ax.set_ylabel(eqobj.SECURITY)


    @classmethod
    def rolling_beta_chart(self, eqobj, ax=None):
        dataVeryShort = eqobj.get_rolling_beta(window = 20).set_index('Date').rename(columns = {'Rolling Beta':'20 days'})
        dataShort = eqobj.get_rolling_beta(window = 60).set_index('Date').rename(columns = {'Rolling Beta':'60 days'})
        dataLong = eqobj.get_rolling_beta(window = 120).set_index('Date').rename(columns = {'Rolling Beta':'120 days'})
        dataVeryShort.plot( ax = ax, grid = True)
        dataShort.plot( ax = ax, grid = True)
        dataLong.plot( ax = ax, grid = True)
        # ax.axhline(0, color = 'black')
        ax.legend(ncol=3, prop={'size':6})
        ax.set_title('Rolling Beta')


    @classmethod
    def rolling_sharpe_ratio_chart(self, eqobj:Equity, ax = None):
        window = 126
        data = eqobj.get_rolling_sharpe_ratio(window = window, risk_free_rate =  0.0)
        data.plot(ax = ax, grid = True)
        ax.set_title(f'Rolling Sharpe {window} days')
        ax.legend(ncol=2, loc='lower left', prop={'size':6})


    def drawdowns_chart(self, eqobj):
        eqobj.get_drawdowns()


    @classmethod
    def monthly_return_heatmap(self, eqobj:Equity, ax = None):
        returns = eqobj.get_returns().reset_index()
        returns['Date'] = pd.to_datetime(returns['Date'], format = "%Y-%m-%d")
        returns.set_index('Date', inplace = True)
        grouped_returns = returns.groupby(pd.Grouper(freq='M')).sum()[eqobj.SECURITY].to_frame()
        grouped_returns['Year'] = grouped_returns.index.strftime('%Y')
        grouped_returns['Month'] = grouped_returns.index.strftime('%b')
        grouped_returns = grouped_returns.pivot('Year', 'Month', eqobj.SECURITY).fillna(0)
        grouped_returns = grouped_returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        grouped_returns *= 100                       
        ax = sns.heatmap(grouped_returns, ax=ax, annot=True, center=0,
                        fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn' )


    @classmethod
    def famma_french_chart(self, eqobj:Equity, ax = None):
        summary, results = eqobj.get_famma_french_attribution()
        results.plot(title=f'Rolling Fama-French 3-Factor model', ax= ax[0])
        ax[0].legend(ncol = 2, prop={'size':6})

        results = results.round(2).reset_index()
        results['Date'] = results['Date'].apply(lambda x : x.strftime('%b'))
        results = results.iloc[-6:]
        render_mpl_table(results, header_columns=0, col_width=2.0, ax=ax[1])


    def trailing_returns_table(self, eqobj:Equity, ax = None):
        results = eqobj.get_trailing_returns()
        render_mpl_table(results, header_columns=0, col_width=2.0, ax=ax)



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Run                                                                                                              │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

fig = plt.figure(figsize=(15, 10), constrained_layout=True) # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html
gs = GridSpec(4, 4, figure=fig)
ax1 = fig.add_subplot(gs[0, : -3])        
ax2 = fig.add_subplot(gs[1, :-3])
ax3 = fig.add_subplot(gs[2, :-3])
ax4 = fig.add_subplot(gs[3, :-3])

ax5 = fig.add_subplot(gs[0, 2:])
ax6 = fig.add_subplot(gs[1, 2:])
ax7 = fig.add_subplot(gs[2, 2:])
ax8 = fig.add_subplot(gs[3, 1:2])
ax9 = fig.add_subplot(gs[3, 2:3])
ax10 = fig.add_subplot(gs[3, 3:])

ax11 = fig.add_subplot(gs[0, 1:2])

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Morningstar Vendor                                                                                               │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
# SECURITY = 'Select Value M' #'UNH'
# BENCHMARK = 'Russell Mid Cap Value TR USD' #'SPY'

# eqobj = Equity(VENDOR = 'Morningstar', SECURITY = SECURITY, BENCHMARK = BENCHMARK,  RETURN_COL = 'Price')
# ts = TearSheet()
# ts.daily_returns_chart(eqobj, ax1)
# ts.cumulative_returns_chart(eqobj, ax2)
# ts.rolling_std_dev_chart(eqobj, ax3)
# ts.beta_chart(eqobj, ax4)
# ts.rolling_beta_chart(eqobj, ax5)
# ts.rolling_sharpe_ratio_chart(eqobj, ax6)
# ts.monthly_return_heatmap(eqobj, ax7)
# ts.famma_french_chart(eqobj, [ax8, ax9])
# ts.daily_return_distribution(eqobj, ax10)
# ts.trailing_returns_table(eqobj, ax11)


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Yahoo Vendor                                                                                                     │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
SECURITY = 'UNH' 
BENCHMARK = 'SPY'
ts = TearSheet()
ts.daily_returns_chart(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2022-09-30", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax1)
ts.cumulative_returns_chart(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2022-10-01", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax2)
ts.rolling_std_dev_chart(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2022-09-30", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax3)
ts.beta_chart(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2022-09-30", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax4)
ts.rolling_beta_chart(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2020-06-30", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax5)
ts.rolling_sharpe_ratio_chart(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2022-01-30", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax6)
ts.monthly_return_heatmap(Equity(SECURITY = SECURITY, BENCHMARK = 'SPY', START_DATE = "2020-01-31", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax7)
ts.famma_french_chart(Equity(SECURITY = SECURITY, BENCHMARK = 'SPY', START_DATE = "2020-01-31", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = [ax8, ax9])
ts.daily_return_distribution(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2022-10-01", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax10)
ts.trailing_returns_table(eqobj = Equity(SECURITY = SECURITY, BENCHMARK = BENCHMARK, START_DATE = "2021-10-01", END_DATE = "2022-12-30", RETURN_COL = 'Adj Close'), ax = ax11)


fig.suptitle(f'{SECURITY}')
plt.savefig('equity_risk_tear_sheet.png')
plt.show()



