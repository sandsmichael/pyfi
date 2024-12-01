import pandas as pd
import numpy as np
from enum import Enum
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from statsmodels.sandbox.regression.predstd import wls_prediction_std
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score

from more_itertools import distinct_permutations as idp
from more_itertools import distinct_combinations as idc


class RegType(Enum):
    UNIVARIATE = 1
    MULTIVARIATE = 2
    PERMUTATIONS = 3
    COMBINATIONS = 4

    

class Regression:
    """ Fit's OLS regression model for a dependent variable based on univariate or multivariate independent variables.

    Standard implementation for modeling a specific linear relationship.
    """

    def __init__(self, df, dep_var):
        
        self.df = df

        self.dep_var = dep_var

        self.indep_var = [v for v in self.df.columns if v != self.dep_var]


    def split(self, test_size:float=0.4) -> None:
        self.df_train, self.df_test = train_test_split(self.df, train_size = 1-test_size, test_size = test_size) 
        
        self.x_train = self.df_train[self.indep_var]
        self.x_test = self.df_test[self.indep_var]

        self.y_train = self.df_train[[self.dep_var]]
        self.y_test = self.df_test[[self.dep_var]]


    def fit(self):

        if hasattr(self, 'df_train'):
            x = self.x_train
            y = self.y_train

        else:
            x = self.df[self.indep_var]
            y = self.df[[self.dep_var]]

        x_with_const = sm.add_constant(x)

        self.model = sm.OLS(y.to_numpy().reshape(-1, 1), x_with_const).fit()

        alpha = self.model.params[0]
        beta = self.model.params[1]
        
        spread = y - (alpha + beta * x)

        f_statistic = self.model.fvalue
        r_squared = self.model.rsquared
        p_values = self.model.pvalues

        return (alpha, beta, spread, f_statistic, r_squared, p_values)
    

    def test(self, plot = False):

        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        self.model = sm.OLS(self.y_train, sm.add_constant(self.x_train)).fit()
    
        y_pred = self.model.predict(sm.add_constant(self.x_test))
        
        mse = mean_squared_error(self.y_test, y_pred)
        r_squared = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r_squared}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}")

        self.y_pred = y_pred

        return self.model  


    def plot_features(self):

        if len(self.indep_var) > 1:
            fig = plt.figure(figsize=(15,8))
            fig = sm.graphics.plot_partregress_grid(self.model, fig=fig)
            plt.show()
        else:
            fig = plt.figure(figsize=(15,8))
            fig = sm.graphics.plot_regress_exog(self.model, self.indep_var, fig=fig)
            plt.show()


    def plot_resid(self):
        fig = plt.figure()
        sns.distplot(self.model.resid, )
        fig.suptitle('Error Terms', fontsize = 20)                   
        plt.xlabel('Errors', fontsize = 18)                         
        plt.show()


    def plot_qq(self):
        stats.probplot(self.model.resid, dist="norm", plot=plt)
        plt.title('QQ Plot of Residuals')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.show()





# class RegressionPairs:
#     """ Fit's a regression model between each combination or permutation of series' contained within a pd.Df. 

#     param: cls
#         An instance of pyfi.core.`TimeSeries`

#     param: how
#         Enumeration of RegType: RegType.COMBINATIONS or RegType.PERMUTATIONS
    
#     """

#     def __init__(self, cls, how):
        
#         self.cls = cls
        
#         self.df = self.cls.df
#         self.how = how

#         if self.how == RegType.PERMUTATIONS:
#             self.pairs = self.get_permutations()
        
#         elif self.how == RegType.COMBINATIONS:
#             self.pairs = self.get_combinations()


#     def get_combinations(self):
#         return pd.DataFrame(idc(self.df.columns, 2), columns=['ts1', 'ts2'])

#     def get_permutations(self):
#         return pd.DataFrame(idp(self.df.columns, 2), columns=['ts1', 'ts2'])


#     @staticmethod
#     def fit(df, row):
#         """ Fit's OLS model between each x,y pair in self.pairs based on data in self.df

#         Calculates spread between regression line and actual observations

#         returns: summary results
#         """
#         x = df[row['ts1']]
#         y = df[row['ts2']]
        
#         x_with_const = sm.add_constant(x.to_numpy().reshape(-1, 1))
#         model = sm.OLS(y.to_numpy().reshape(-1, 1), x_with_const).fit()

#         alpha = model.params[0]
#         beta = model.params[1]
        
#         spread = y - (alpha + beta * x)  # NOTE: important!

#         f_statistic = model.fvalue
#         r_squared = model.rsquared
#         p_values = model.pvalues

#         return (alpha, beta, spread, f_statistic, r_squared, p_values)


#     def model(self):
#         """ Fits linear regression model according to `how` param and returns summary statistics. 

#         Test regression spread for stationarity.
#         """
#         self.reg_results = self.pairs.apply(lambda row : self.fit(self.df, row), axis=1)

#         df_permut = self.pairs.copy()

#         df_permut['alpha'] = self.reg_results.apply(lambda x: x[0]).round(2)
#         df_permut['beta'] = self.reg_results.apply(lambda x: x[1]).round(2)
#         df_permut['f_statistic'] =self.reg_results.apply(lambda x: x[3]).round(2)
#         df_permut['r_squared'] = self.reg_results.apply(lambda x: x[4]).round(2)
#         df_permut[['p_value_intercept', 'p_value']] = self.reg_results.apply(lambda x: pd.Series(x[5])).round(2)
#         df_permut['id'] = df_permut['ts1'].astype(str) + '_' + df_permut['ts2'].astype(str)

#         return df_permut
    

#     def get_spread(self):
#         # regression spread
#         spread = self.pairs.merge(self.reg_results.apply(lambda x: x[2]), how = 'outer', left_index = True, right_index = True)
#         spread['id'] = spread.ts1 + '_' + spread.ts2
#         spread.drop(columns = ['ts1','ts2'], inplace = True)
#         spread.set_index('id', inplace = True)
#         spread = spread.T
#         self.spread = spread

#         return spread


#     def get_spread_z_score(self):
#         return (self.spread - self.spread.mean()) / self.spread.std()


#     def get_summary(self):
#         summary = self.model()
#         spread = self.get_spread().reset_index().melt(id_vars='index', var_name = 'id').rename(columns={'index':'date'})
#         spread_z = self.get_spread_z_score().reset_index().melt(id_vars='index', var_name = 'id').rename(columns={'index':'date'})

#         return summary, spread, spread_z
    