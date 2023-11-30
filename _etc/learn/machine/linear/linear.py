import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class Regression():
    """Pass an unindexed dataframe as data
    To use multivariate regression; Do not pass an indep_var.
    """    
    def __init__(self, data:pd.DataFrame, dep_var:str, indep_var:str=None, verbose = False) -> None:
        self.df = data
        self.dep_var = dep_var

        if indep_var != None:
            self.how = 'univariate'
            self.indep_var = indep_var
        else:
            self.how = 'multivariate'
            self.indep_var = [c for c in self.df if c != self.dep_var]

        if verbose:
            print(self.how)
            print(self.df.info())
            print(self.df.describe())
            print(self.df.head())


    def examine_stats(self):
        ''' Examine the historical relationship between features using statsmodels OLS.
        '''
        df = self.df.copy()
        
        y_train = df[self.dep_var]
        self.y_train = y_train
        
        df.pop(self.dep_var)
        X_train = df
        self.X_train = X_train

        if self.how == 'univariate':
            X_train_lm = sm.add_constant(X_train[self.indep_var]) 
        
        else:
            X_train_lm = sm.add_constant(X_train)  # Multiple regression

        self.model = sm.OLS(y_train.astype(float), X_train_lm.astype(float)).fit() # OLS Linear Regression
        
        model_summary = self.model.summary()

        return self.model


    def plot_linear_univariate(self):
        ''' Plot linear relationship between two features for a univariate model.
        '''
        plt.figure(figsize = (10, 6))
        plt.rcParams.update({'font.size': 14})
        plt.xlabel(self.indep_var)
        plt.ylabel(self.dep_var)
        plt.title("Simple linear regression model")
        plt.scatter(self.df[self.indep_var], self.df[self.dep_var])
        plt.plot(self.df[self.indep_var], self.model.params.loc['const'] + self.model.params.loc[self.indep_var] * self.df[self.indep_var],
                label='Y={:.4f}+{:.4f}X'.format(self.model.params.loc['const'], self.model.params[self.indep_var]), 
                color='red')
        plt.legend()
        plt.show()


    def reg_plots(self):
        if self.how == 'univariate':

            fig = plt.figure(figsize=(15,8))
            fig = sm.graphics.plot_regress_exog(self.model, self.indep_var, fig=fig)
            plt.show()
        
        elif self.how == 'multivariate':

            fig = plt.figure(figsize=(15,8))
            fig = sm.graphics.plot_partregress_grid(self.model, fig=fig)
            plt.show()
 

    def confidence_intervals(self):
        x = self.df[self.indep_var]
        y = self.df[self.dep_var]
        _, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(self.model)
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(x, y, 'o', label="data")
        ax.plot(x, self.model.fittedvalues, 'g--.', label="OLS")
        ax.plot(x, confidence_interval_upper, 'r--')
        ax.plot(x, confidence_interval_lower, 'r--')
        ax.legend(loc='best')
        plt.xlabel(self.indep_var)
        plt.ylabel(self.dep_var)
        plt.show()
        

    def get_params(self):
        p = self.model.pvalues.to_frame()
        p.columns = ['pvalues']

        coefs = self.model.params.to_frame()
        coefs.columns = ['coefs']

        return pd.concat([p, coefs], axis=1)


    # Scikit Learn
    def train_model(self, test_size = 0.2, SEED = None):

        if self.how == 'univariate':
            X = self.df[self.indep_var].values.reshape(-1, 1)
            y = self.df[self.dep_var].values.reshape(-1, 1)
        else:
            X = self.df[self.indep_var]
            y = self.df[self.dep_var]     

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = SEED)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        # print(regressor.intercept_)
        # print(regressor.coef_)

        self.regressor = regressor
        self.X_test = X_test
        self.y_test = y_test

        return regressor


    def predict(self):
        y_pred = self.regressor.predict(self.X_test)
        self.y_pred = y_pred
        return pd.DataFrame({'Actual': self.y_test.squeeze(), 'Predicted': y_pred.squeeze()})


    def evaluate_predictions(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)

        print(f'Mean absolute error: {mae:.2f}')
        print(f'Mean absolute percentage error: {mape:.2f}')
        print(f'Mean squared error: {mse:.2f}')
        print(f'Root mean squared error: {rmse:.2f}')

        return mae, mape, mse, rmse


    def plot_residuals(self):
        fig = plt.figure()
        sns.distplot((self.y_test - self.y_pred), )
        fig.suptitle('Error Terms', fontsize = 20)                   
        plt.xlabel('Errors', fontsize = 18)                         
        plt.show()


################################################################################################
# NOTE: Deprecate below. 
# Out of sample prediction with statsmodels
def sm_oos_predict(self, X_new:list = None, most_recent:bool = False):
    ''' X_new is a series in the same shape of the X_train data
    X_new[self.get_numeric_cols()] = self.scaler.transform(X_new[self.get_numeric_cols()]) 
    df = pd.DataFrame(self.scaler.inverse_transform(X_new))
    ''' 
    if most_recent:
        mr = self.df.drop(self.dep_var, axis=1)
        X_new = pd.DataFrame([1] + mr.iloc[-1].values.tolist()).transpose() 

    else:
        X_new = pd.DataFrame(X_new).transpose() 

    X_test_lm_new = sm.add_constant(X_new)
    y_pred_new = self.model.predict(X_test_lm_new)
    X_new.columns = ['const'] + [c for c in self.df_train.columns if c != self.dep_var]
    print(X_new)
    print(y_pred_new)
    return y_pred_new.values
# Test set predictions with statsmodels
def test_model(self):
    df_test = self.df_test
    y_test = df_test[self.dep_var]
    
    df_test.pop(self.dep_var)
    X_test = df_test
    X_test_lm = sm.add_constant(X_test)
    
    y_pred = self.model.predict(X_test_lm)
    # print(r2_score(y_true = y_test, y_pred = y_pred))

