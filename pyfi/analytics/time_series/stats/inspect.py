import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
        res = res.reset_index().rename(columns = {'index':'id'})
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


    def eig(self):
        '''numpy.linalg.eig function to decompose the symmetric covariance matrix into its eigenvalues and eigenvectors
        '''
        cov_mat = self.df.cov()

        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        
        return cov_mat, eigen_vals, eigen_vecs

    def explained_variance(self, plot = False):
        
        cov_mat, eigen_vals, eigen_vecs = self.eig()

        eigen_vals = pd.DataFrame(eigen_vals)
        print(eigen_vals)
                
        var_exp = eigen_vals.divide(eigen_vals.sum()) #[(i / tot) for i in sorted(eigen_vals, reverse=True)]
        
        cum_var_exp = var_exp.cumsum()

        # if plot:
        #     plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        #             label='Individual explained variance')
        #     plt.step(range(1, 14), cum_var_exp, where='mid',
        #             label='Cumulative explained variance')
        #     plt.ylabel('Explained variance ratio')
        #     plt.xlabel('Principal component index')
        #     plt.legend(loc='best')
        #     plt.tight_layout()
        #     plt.show()
        
        return cum_var_exp


    def vif(self):
        """[Summary]
        Variance Inflation Factor or VIF is a quantitative value that says how much the feature variables are correlated with each other. 
        Keep varibles with VIF values < 5
        If VIF > 5 and high p-value, drop the variable; Rinse and repeat until all variables have VIF < 5 and significant p-values (<0.005)
        """        
        vif = pd.DataFrame()
        vif['Features'] = self.df.columns
        vif['VIF'] = [variance_inflation_factor(self.df.values, i) for i in range(self.df.shape[1])]
        vif['VIF'] = round(vif['VIF'], 2)
        vif = vif.sort_values(by = "VIF", ascending = False).to_dict()

        return vif

    # def get_cat_feats(data=None):
    #     '''
    #     Returns the categorical features in a data set
    #     '''
    #     if data is None:
    #         raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    #     cat_features = data.select_dtypes(include=['object']).columns

    #     return list(cat_features)


    # def get_num_feats(data=None):
    #     '''
    #     Returns the numerical features in a data set
    #     '''
    #     if data is None:
    #         raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    #     num_features = data.select_dtypes(exclude=['object', 'datetime64']).columns

    #     return list(num_features)

            

    # def display_missing(data=None, plot=False):
    #     '''
    #     Display missing values as a pandas dataframe.
    #     '''
    #     if data is None:
    #         raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    #     df = data.isna().sum()
    #     df = df.reset_index()
    #     df.columns = ['features', 'missing_counts']

    #     missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
    #     df['missing_percent'] = missing_percent

    #     if plot:
    #         sns.heatmap(data.isnull(), cbar=True)
    #         plt.show()
    #         return df
    #     else:
    #         return df


    # def detect_outliers(data, n, features):
    #     '''
    #         Detect Rows with outliers.
    #         Parameters
    #         '''
    #     outlier_indices = []

    #     # iterate over features(columns)
    #     for col in features:
    #         # 1st quartile (25%)
    #         Q1 = np.percentile(data[col], 25)
    #         # 3rd quartile (75%)
    #         Q3 = np.percentile(data[col], 75)
    #         # Interquartile range (IQR)
    #         IQR = Q3 - Q1

    #         # outlier step
    #         outlier_step = 1.5 * IQR

    #         # Determine a list of indices of outliers for feature col
    #         outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index

    #         # append the found outlier indices for col to the list of outlier indices
    #         outlier_indices.extend(outlier_list_col)

    #     # select observations containing more than 2 outliers
    #     outlier_indices = Counter(outlier_indices)
    #     multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    #     return multiple_outliers

