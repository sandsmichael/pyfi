import pandas as pd
import numpy as np
from pyfi.lib.datasci.machine_learning.regression import RegressionPairs, RegType

from more_itertools import distinct_permutations as idp
import statsmodels.tsa.stattools as st
from statsmodels.tsa.vector_ar.vecm import coint_johansen



class Pairs:
    """ Statistical analysis conducted across each permutation of of features included in the dataframe. """

    def __init__(self, df):
        self.df = df

        ts = TimeSeries(df=df)

        stationarity = ts.get_stationarity()
        coef, pval, tall = ts.get_correlation()
        coef, pval, tall = ts.get_cointegration()
        ratio, z_score = ts.get_feature_ratios()
        summary, spread, spread_z, spread_adf = ts.get_regression_spread()


    def get_summary():
        pass

# def run_backtest(S1, S2, window1, window2):

#     # If window length is 0, algorithm doesn't make sense, so exit
#     if (window1 == 0) or (window2 == 0):
#         return 0
    
#     # Compute rolling mean and rolling standard deviation
#     ratios = S1/S2
#     ma1 = ratios.rolling(window=window1,
#                                center=False).mean()
#     ma2 = ratios.rolling(window=window2,
#                                center=False).mean()
#     std = ratios.rolling(window=window2,
#                         center=False).std()
#     zscore = (ma1 - ma2)/std
#     # print(zscore)
    
#     # Simulate trading
#     money = 0
#     countS1 = 0
#     countS2 = 0
#     for i in range(len(ratios)):
#         # Sell short if the z-score is > 1
#         if zscore[i] < -1:
#             money += S1[i] - S2[i] * ratios[i]
#             countS1 -= 1
#             countS2 += ratios[i]
#             # print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
#         # Buy long if the z-score is < -1
#         elif zscore[i] > 1:
#             money -= S1[i] - S2[i] * ratios[i]
#             countS1 += 1
#             countS2 -= ratios[i]
#             # print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
#         # Clear positions if the z-score between -.5 and .5
#         elif abs(zscore[i]) < 0.75:
#             money += S1[i] * countS1 + S2[i] * countS2
#             countS1 = 0
#             countS2 = 0
#             # print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
            
#     return money

# # trade(prices['AAPL'], prices['MSFT'], 60, 5)


# def run_price_ratio_strategy(df):
    
#     print('***********RUNNING PRICE RATIO STRATEGY***********')
    
#     ratio_ts, ratio_z_score_ts = run_price_ratio(df)
    
#     # Rolling average of pair price ratio z-score
#     ratio_z_score_ma_5 = ratio_z_score_ts.rolling(window=5, center=False).mean()
#     ratio_z_score_ma_20 = ratio_z_score_ts.rolling(window=20, center=False).mean()
#     ratio_z_score_ma_60 = ratio_z_score_ts.rolling(window=60, center=False).mean()
#     ratio_z_score_ma_stdev_60 = ratio_z_score_ts.rolling(window=60, center=False).std()
#     zscore_60_5 = (ratio_z_score_ma_5 - ratio_z_score_ma_60) / ratio_z_score_ma_stdev_60

#     if len(df.columns) < 25:
#         plt.figure(figsize=(12, 6))
#         plt.plot(pd.to_datetime(zscore_60_5.index), zscore_60_5.values)
#         plt.legend(zscore_60_5.columns)

#         plt.ylabel('Ratio')
#         plt.axhline(0, color='black')
#         plt.axhline(1.0, color='black', linestyle='--')
#         plt.axhline(-1.0, color='black', linestyle='--')    
#         plt.show()
    
#     buy = zscore_60_5.copy()
#     sell = zscore_60_5.copy()
    
#     for c in buy.columns:
#         buy[c][buy[c]>-1] = 0

#     for c in sell.columns:
#         sell[c][sell[c]<1] = 0
    
#     # display(buy.tail())
#     # display(sell.tail())

#     print('***********CALCULATING PRICE RATIO STRATEGY P/L***********')
    
#     df_permut = pd.DataFrame(idp(df.columns, 2), columns=['ts1', 'ts2'])
#     def calculate_pl(x):
#         result = run_backtest( df[x['ts1']], df[x['ts2']], 60, 5)
#         return result
#     df_permut = df_permut.apply(calculate_pl, axis=1)
#     df_permut.index = ratio_ts.columns
    
    
#     return zscore_60_5, buy, sell, df_permut, ratio_ts, ratio_z_score_ts

# # run_price_ratio_strategy(prices)
