from pyfi.lib.datasci.features.features import Features
from pyfi.lib.datasci.time_series.stats.descriptive import Descriptive, Significance
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
import statsmodels.api as sm


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

    def get_stationarity(self, alpha = 0.05):
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

            result[col] = {'adf_stat': adf_result[0], 'p_value': adf_result[1], 'lags': adf_result[2], 'n-obs': adf_result[3]}
            # for key, value in adf_result[4].items():
            #     result[col][f'Critical Value {key}'] = value

        res = pd.DataFrame(result).transpose()
        res = Significance().add_significance_levels(df=res)
        res = res.reset_index().rename(columns = {'index':'id'})
        return res


    def get_correlation(self, significance_stars=False):
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
        

        def get_correlation_summary():
            data = self.df
            results = []

            variables = data.columns
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i <= j:  # Only compute for unique pairs (including diagonal)
                        corr, pval = pearsonr(data[var1], data[var2])
                        results.append({
                            'feature_1': var1,
                            'feature_2': var2,
                            'coef_pearson': corr,
                            'p_value': pval if var1 != var2 else np.nan  # p-value only for off-diagonal
                        })
            return pd.DataFrame(results)

        if significance_stars:
            coef = Significance().annotate_floats_with_stars(label_df=coef, pval_df=pvalues)
            pvalues = Significance().annotate_floats_with_stars( label_df=pvalues, pval_df=pvalues)

        summary = Significance().add_significance_levels(df=get_correlation_summary())
        return coef, pvalues, summary


    def get_cointegration(self, significance_stars=False):
        """
        Calculate cointegration test statistic matrix for all pairs of columns
        
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.JohansenTestResult.html 
        """
        dfcols = self.df.columns
        coef = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)
        pvalues = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)

        for r in dfcols:
            for c in dfcols:
                if r == c:
                    # Diagonal values should be NaN or 0 (self-cointegration is undefined)
                    coef.loc[r, c] = np.nan
                    pvalues.loc[r, c] = np.nan
                else:
                    tmp = self.df[[r, c]].dropna()
                    test_stat, pval, crit = coint(tmp[r], tmp[c])
                    
                    coef.loc[r, c] = round(test_stat, 6)
                    pvalues.loc[r, c] = round(pval, 6)

        def get_cointegration_summary():
            data = self.df
            results = []

            variables = data.columns
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i <= j:  # Only compute for unique pairs (including diagonal)
                        if var1 == var2:
                            test_stat, pval, crit = np.nan, np.nan, np.nan
                        else:
                            test_stat, pval, crit = coint(data[var1], data[var2])
                        
                        results.append({
                            'feature_1': var1,
                            'feature_2': var2,
                            't_stat': test_stat,
                            'p_value': pval if var1 != var2 else np.nan  # p-value only for off-diagonal
                        })

            return pd.DataFrame(results)
        
        if significance_stars:
            coef = Significance().annotate_floats_with_stars(label_df=coef, pval_df=pvalues)     
            pvalues = Significance().annotate_floats_with_stars( label_df=pvalues, pval_df=pvalues)

        summary = Significance().add_significance_levels(df=get_cointegration_summary())
        return coef, pvalues, summary


    def get_regression(self, significance_stars=False):
        """
        Fit linear regressions between each pair of columns in the DataFrame using statsmodels.
        Return regression coefficients, p-values, F-statistics, and a tall-format table.

        Args:
            significance_stars (bool): Whether to annotate significance levels.

        Returns:
            coef_matrix (pd.DataFrame): Matrix of regression coefficients.
            pvalues_matrix (pd.DataFrame): Matrix of p-values.
            fstat_matrix (pd.DataFrame): Matrix of F-statistics.
            tall (pd.DataFrame): Tall-format table of regression results.
        """
        dfcols = self.df.columns
        coef = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)
        pvalues = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)
        fstat = pd.DataFrame(index=dfcols, columns=dfcols, dtype=float)

        for r in dfcols:
            for c in dfcols:
                if r == c:
                    # Diagonal values should be NaN
                    coef.loc[r, c] = np.nan
                    pvalues.loc[r, c] = np.nan
                    fstat.loc[r, c] = np.nan
                else:
                    tmp = self.df[[r, c]].dropna()
                    X = tmp[r].values  # Independent variable
                    y = tmp[c].values  # Dependent variable
                    x = sm.add_constant(X)  # Add intercept term

                    model = sm.OLS(y, x).fit()

                    coef.loc[r, c] = model.params[1]  # Slope (coefficient for x)
                    pvalues.loc[r, c] = model.pvalues[1]  # P-value for slope
                    fstat.loc[r, c] = model.fvalue  # F-statistic

        def get_summary():
            """
            Generate a tall-format DataFrame with regression results.
            """
            results = []
            for i, var1 in enumerate(dfcols):
                for j, var2 in enumerate(dfcols):
                    if i < j:  # Exclude diagonal and redundant pairs
                        tmp = self.df[[var1, var2]].dropna()
                        x = tmp[var1].values
                        y = tmp[var2].values
                        x = sm.add_constant(x)

                        model = sm.OLS(y, x).fit()
                        results.append({
                            'feature_1': var1,
                            'feature_2': var2,
                            'coef_regression': model.params[1],
                            'intercept': model.params[0],
                            'p_value': model.pvalues[1],
                            'r_squared': model.rsquared,
                            'f_statistic': model.fvalue
                        })
            return pd.DataFrame(results)

        if significance_stars:
            coef = Significance().annotate_floats_with_stars(label_df=coef, pval_df=pvalues)     
            pvalues = Significance().annotate_floats_with_stars( label_df=pvalues, pval_df=pvalues)

        summary = Significance().add_significance_levels(df=get_summary())
        return coef, pvalues, fstat, summary


    def get_regression_spread(self):
        dfcols = self.df.columns
        spread_df = pd.DataFrame(index=self.df.index)  # Initialize spread_df with the same index as df

        for i, r in enumerate(dfcols):
            for j, c in enumerate(dfcols):
                if r == c:
                    # Diagonal values should be NaN, no regression for self-pairing
                    continue
                elif i<j :
                    tmp = self.df[[r, c]].dropna()  # Drop rows with NaN in either column
                    X = tmp[r].values  # Independent variable
                    y = tmp[c].values  # Dependent variable
                    x = sm.add_constant(X)  # Add intercept term to independent variable

                    model = sm.OLS(y, x).fit()

                    alpha = model.params[0]  # Intercept
                    beta = model.params[1]   # Slope
                    spread = y - (alpha + beta * X)  # Spread (residuals)

                    spread_df[f'{r}_vs_{c}_spread'] = pd.Series(spread, index=tmp.index)


        spread_z_score = Features(df=spread_df).standardize()
        spread_adf = TimeSeries(df=spread_df.dropna(how='any',axis=0)).get_stationarity()
        return spread_df, spread_z_score, spread_adf



    def get_feature_ratios(self):
        ratio, z_score = self.features.get_feature_ratios()
        
        adf = TimeSeries(df=ratio.dropna(how='any',axis=0)).get_stationarity() # test if ratio is stationary

        return ratio, z_score, adf


    # def get_regression_spread(self):
    #     from pyfi.lib.datasci.machine_learning.regression import RegressionPairs, RegType
    #     regr = RegressionPairs(cls = self, how = RegType.COMBINATIONS)
    #     summary, spread, spread_z = regr.get_summary()
    #     adf = TimeSeries(df=spread.pivot(index='date', columns = 'id', values = 'value').dropna(how='any',axis=0)).get_stationarity() # test if spread is stationary
    #     return summary, spread, spread_z, adf


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





