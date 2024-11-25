from pyfi.lib.datasci.features.features import Features
from pyfi.lib.datasci.time_series.stats.descriptive import Descriptive
# from pyfi.lib.datasci.time_series.technical_analysis.ta import TechnicalAnalysis
import pandas as pd
import numpy as np
from enum import Enum
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint
from scipy import stats


class Frequency(Enum):
    DAILY = 'D'
    WEEKLY_M = 'W-MON'
    WEEKLY_F = 'W-FRI'
    MONTHLY = 'M'

class AggFunc(Enum):
    MEAN = np.mean
    MEDIAN = np.median
    FIRST = 'first'
    LAST = 'last'


class TimeSeries():

    def __init__( self,
                  df:pd.DataFrame,
                  dep_var:str = None,
                  indep_var:list = None,
                 ) -> None:
        """ Operates on a date/time indexed pandas DataFrame
        
        """
        # super().__init__(df=df)

        self.df = df
        self.dep_var = dep_var
        self.indep_var = indep_var

        self.df.index.name = 'dt'

        if indep_var is not None:
            self.df = self.df[[dep_var, *indep_var]]

        self.num_cols = self.features.get_num_feats()
        self.cat_cols = self.features.get_cat_feats()

        # self.ta = TechnicalAnalysis()

        # TODO: Check/Handle missing/null
        # TODO: Check/Handle look ahead bias

    ##############################
    """ Use Property to set self.features when any changes are applied to self.df
    self.df represents the root dataset being used by TimeSeries. 
    Since transformations can be applied directly onto it, getter/setter logic ensures partity between the TimeSeries.df var and Features.df.
    # This ensures that any use of Features methods applies to the latest instance of TimeSeries.df
    """
    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, updated_df):
        self._df = updated_df
        self._features = self.update_features_instance_var(updated_df)

    @property
    def features(self):
        return self._features
    
    def update_features_instance_var(self, updated_df):
        return Features(df=updated_df, dep_var = None)
    ##############################

    def group(self, frequency, aggfunc):
        """ Group by and aggregate
        """
        if frequency is None or aggfunc is None: raise ValueError ('Error - neither frequency nor aggfunc can be null')
        
        if not isinstance(aggfunc.value, str):
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).apply(aggfunc)
        elif aggfunc == AggFunc.LAST:
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).last()
        elif aggfunc == AggFunc.FIRST:
            self.df = self.df.groupby(pd.Grouper(freq=frequency.value)).first()
        
        else: raise ValueError('Error - aggfunc parameter is not valid.')
        
        
    def winsorize(self, subset = None, limits = [.05, .05]):
        self.df = self.features.winsorize_columns(df = self.df, subset = subset, limits=limits)


    def standardize(self):
        self.df = self.features.standard_scaler(df = self.df)

    
    def de_standardize(self):
        self.df = self.features.inverse_standard_scaler(df = self.df)

    def naive_transform(self):
        transformed_df, transformations_applied = self.features.naive_transform()
        self.df = transformed_df
        return transformed_df, transformations_applied

    def normalize(self):
        pass

    def de_normalize(self):
        pass

    def get_stationarity(self, df = None, alpha = 0.05):
        """ Test for Stationarity using Augmented Dicky Fuller Test
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
            # for key, value in adf_result[4].items():
            #     result[col][f'Critical Value {key}'] = value

        res = pd.DataFrame(result).transpose()
        res['bool'] = [True if x <= alpha else False for x in res['p-value']]
        res = res.reset_index().rename(columns = {'index':'id'})
        return res


    def get_correlation(self):
        """
        Calculate P-value matrix associated with Pearson Correlation
        """
        coef = self.df.corr(method='pearson')

        dfcols = self.df.columns
        pvalues = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)

        for r in dfcols:
            for c in dfcols:
                if r == c:
                    # Diagonal values should be 0 (self-correlation has no p-value)
                    pvalues.loc[r, c] = 0.0
                else:
                    # Filter rows with non-null values in both columns
                    tmp = self.df[[r, c]].dropna()
                    # Compute the p-value using pearsonr
                    _, pval = pearsonr(tmp[r], tmp[c])
                    pvalues.loc[r, c] = round(pval, 6)  
        

        def get_correlation_tall():
            data = self.df
            results = []

            variables = data.columns
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i <= j:  # Only compute for unique pairs (including diagonal)
                        corr, pval = pearsonr(data[var1], data[var2])
                        results.append({
                            'Feature_1': var1,
                            'Feature_2': var2,
                            'Coefficient': corr,
                            'P_Value': pval if var1 != var2 else np.nan  # p-value only for off-diagonal
                        })
            return pd.DataFrame(results)


        return coef, pvalues, get_correlation_tall()


    def get_cointegration(self):
        """
        Calculate cointegration test statistic matrix for all pairs of columns
        
        #         https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
        #         https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.JohansenTestResult.html 
        #         
        #         # NOTE: models produce output with multiple arrays for different rank tests. Hardcoded [0] refers to intercept, 1 refers to coef..        
        """
        dfcols = self.df.columns
        coint_stats = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)
        pvalues = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)

        for r in dfcols:
            for c in dfcols:
                if r == c:
                    # Diagonal values should be NaN or 0 (self-cointegration is undefined)
                    coint_stats.loc[r, c] = np.nan
                    pvalues.loc[r, c] = np.nan
                else:
                    tmp = self.df[[r, c]].dropna()
                    # Perform the cointegration test
                    test_stat, pval, crit = coint(tmp[r], tmp[c])
                    
                    coint_stats.loc[r, c] = round(test_stat, 6)
                    pvalues.loc[r, c] = round(pval, 6)


        def get_cointegration_tall():
            data = self.df
            results = []

            variables = data.columns
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i <= j:  # Only compute for unique pairs (including diagonal)
                        corr, pval = pearsonr(data[var1], data[var2])
                        results.append({
                            'Feature_1': var1,
                            'Feature_2': var2,
                            'Coefficient': corr,
                            'P_Value': pval if var1 != var2 else np.nan  # p-value only for off-diagonal
                        })

            return pd.DataFrame(results)

        return coint_stats, pvalues, get_cointegration_tall()


    def get_feature_ratios(self):
        ratio, z_score = self.features.get_feature_ratios()
        
        adf = TimeSeries(df=ratio.dropna(how='any',axis=0)).get_stationarity() # test if ratio is stationary

        return ratio, z_score, adf


    def get_regression_spread(self):
        from pyfi.lib.datasci.machine_learning.regression import RegressionPairs, RegType
        print(self.df)
        regr = RegressionPairs(cls = self, how = RegType.PERMUTATIONS)
        summary, spread, spread_z = regr.get_summary()
        adf = TimeSeries(df=spread.pivot(index='date', columns = 'id', values = 'value').dropna(how='any',axis=0)).get_stationarity() # test if spread is stationary
        return summary, spread, spread_z, adf


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
        
        return df_decomposed


    def describe(self):
        return Descriptive(data = self.df).describe()


    def summary():
        # Generic summary of inspected data
        pass


# def plot_seasonal_decompose(self):
#     plt.figure(figsize=(10, 8))

#     plt.subplot(411)
#     plt.plot(data, label='Original Time Series')
#     plt.legend()

#     plt.subplot(412)
#     plt.plot(decomposition.trend, label='Trend')
#     plt.legend()

#     plt.subplot(413)
#     plt.plot(decomposition.seasonal, label='Seasonal')
#     plt.legend()

#     plt.subplot(414)
#     plt.plot(decomposition.resid, label='Residuals')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()





