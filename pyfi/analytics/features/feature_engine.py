# from pyfi.core.timeseries import TimeSeries

import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from dateutil.parser import parse
import re



class FeatureEngine:
    """ feature engineering and selection: Generates feature sets that maximize the explained variance of the dep var.
    """

    def __init__(self, df, dep_var) -> None:
        
        super().__init__(
            df,
            dep_var=None,
        )

        self.curate() # self.df is assigned here

        self.dep_var = dep_var
        


class Processor():
    """ Functionality for cleaning and pre-processing raw features
    """

    def __init__(
            self,
            df,
            dep_var=None,
        ) -> None:
        
        self.df = df

        self.dep_var = dep_var
        

    def get_numeric_cols(self):
        return self.df.select_dtypes(exclude=['object', 'datetime64']).columns

    def cast_numeric_cols(self):
        for c in self.df.columns:
            if c in self.get_numeric_cols():
                self.df[c] = pd.to_numeric(self.df[c])

    def get_cat_feats(self):
        '''
        Returns the categorical features in a data set
        '''
        return list(self.df.select_dtypes(include=['object']).columns)


    def get_num_feats(self):
        '''
        Returns the numerical features in a data set
        '''
        return list(self.df.select_dtypes(exclude=['object', 'datetime64']).columns)


    def curate(self):
        return self.df.dropna(how = 'any', axis=0) # detect and cast datatypes.


    def winsorize_columns(self, df, subset = None, limits=[0.05, 0.05]):
        winsorized_df = df.copy()

        if subset is None:
            subset = winsorized_df.columns

        for column in subset:
            winsorized_df[column] = winsorize(df[column], limits=limits)

        return winsorized_df
    

    def standard_scaler(self, df):
        """ Standardize each series of a dataframe by (observation - mean) /  standard deviation 
        """
        self.scaler = StandardScaler()
        self.scaled_values = self.scaler.fit_transform(df)
        return pd.DataFrame(self.scaled_values, columns=df.columns)
        

    def inverse_standard_scaler(self, df):
        unscaled_values = self.scaler.inverse_transform(df)
        return pd.DataFrame(unscaled_values, columns=df.columns)


    def binary_encode(self, var_list:str) -> None:
        def binary_map(x):
            return x.map({'yes': 1, "no": 0})
        self.df[var_list] = self.df[var_list].apply(binary_map)


    def cat_encode(self, var_list:str) -> None:
        for var in var_list:
            dummies = pd.get_dummies(self.df[var])
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(var, axis=1, inplace=True)


    def convert_dtypes(self, df):
        ''' Convert datatype of a feature to its original datatype.
        If the datatype of a feature is being represented as a string while the initial datatype is an integer or a float 
        or even a datetime dtype. The convert_dtype() function iterates over the feature(s) in a pandas dataframe and convert the features to their appropriate datatype
        '''
        if df.isnull().any().any() == True:
            raise ValueError("DataFrame contain missing values")
        else:
            i = 0
            changed_dtype = []

            def is_date(string, fuzzy=False):
                try:
                    parse(string, fuzzy=fuzzy)
                    return True
                except ValueError:
                    return False
                
            while i <= (df.shape[1])-1:
                val = df.iloc[:,i]
                if str(val.dtypes) =='object':
                    val = val.apply(lambda x: re.sub(r"^\s+|\s+$", "",x, flags=re.UNICODE)) #Remove spaces between strings
            
                try:
                    if str(val.dtypes) =='object':
                        if val.min().isdigit() == True: #Check if the string is an integer dtype
                            int_v = val.astype(int)
                            changed_dtype.append(int_v)
                        elif val.min().replace('.', '', 1).isdigit() == True: #Check if the string is a float type
                            float_v = val.astype(float)
                            changed_dtype.append(float_v)
                        elif is_date(val.min(),fuzzy=False) == True: #Check if the string is a datetime dtype
                            dtime = pd.to_datetime(val)
                            changed_dtype.append(dtime)
                        else:
                            changed_dtype.append(val) #This indicate the dtype is a string
                    else:
                        changed_dtype.append(val) #This could count for symbols in a feature
                
                except ValueError:
                    raise ValueError("DataFrame columns contain one or more DataType")
                except:
                    raise Exception()

                i = i+1

            data_f = pd.concat(changed_dtype,1)

            return data_f




class Generator(FeatureEngine):
    """ Generates new features from provided dataset
    """
    def create_polynomial_features(self, degree=2):
            """
            Create polynomial features for the given DataFrame.
            
            Args:
            - df: DataFrame containing predictor variables
            - degree: Degree of polynomial features to create (default is 2)
            
            Returns:
            - DataFrame with polynomial features
            """
            poly = PolynomialFeatures(degree=degree)
            poly_features = poly.fit_transform(self.df)
            col_names = [f"Poly_{i}" for i in range(poly_features.shape[1])]
            self.df[col_names] = poly_features
            return self.df


        # def add_technical_indicators(self):

        #     for col in self.df.columns:
        #         self.df[f'{col}_rsi'] = self.add_rsi(self.df[col])

        #     pass
    


class Selector(FeatureEngine):
    """ Selects the best features to include in the model
    """

    def get_explained_variance(self):

        target_column_name = self.dep_var
        target_column = self.df[target_column_name]

        r2_scores = {}

        for column in self.df.columns:
            if column != target_column_name:
                predictor_column = self.df[column].values.reshape(-1, 1)
                
                model = LinearRegression()
                model.fit(predictor_column, target_column)
                
                r2_scores[column] = model.score(predictor_column, target_column)

        return r2_scores


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
  

    def correlation_feature_selection(self, threshold=0.5):
        """
        Perform feature selection based on correlation with the target variable.
        
        Args:
        - df: DataFrame containing predictor and target variables
        - target_column: Name of the target variable column
        - threshold: Correlation threshold value
        
        Returns:
        - List of selected feature names based on correlation
        """
        corr_matrix = self.df.corr()
        selected_features = corr_matrix[abs(corr_matrix[self.dep_var]) > threshold][self.dep_var].index.tolist()
        selected_features.remove(self.dep_var)
        return selected_features

    def rfe_feature_selection(self, n_features_to_select=5):
        """
        Perform feature selection using Recursive Feature Elimination (RFE).
        
        Args:
        - df: DataFrame containing predictor and target variables
        - target_column: Name of the target variable column
        - n_features_to_select: Number of features to select
        
        Returns:
        - List of selected feature names based on RFE
        """
        X = self.df.drop(columns=[self.dep_var])
        y = self.df[self.dep_var]
        model = LinearRegression()
        rfe = RFE(model)
        rfe = rfe.fit(X, y)
        selected_features = list(X.columns[rfe.support_])
        return selected_features

    def select_k_best_feature_selection(self, k=5):
        """
        Perform feature selection using SelectKBest.
        
        Args:
        - df: DataFrame containing predictor and target variables
        - target_column: Name of the target variable column
        - k: Number of top features to select
        
        Returns:
        - List of selected feature names based on SelectKBest
        """
        X = self.df.drop(columns=[self.dep_var])
        y = self.df[self.dep_var]
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = list(X.columns[selector.get_support()])
        return selected_features


    def lasso_feature_selection(self, alpha=1.0):
        """
        Perform feature selection using Lasso Regression.
        
        Args:
        - df: DataFrame containing predictor and target variables
        - target_column: Name of the target variable column
        - alpha: Regularization strength
        
        Returns:
        - List of selected feature names based on Lasso Regression
        """
        X = self.df.drop(columns=[self.dep_var])
        y = self.df[self.dep_var]
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        selected_features = list(X.columns[lasso.coef_ != 0])
        return selected_features


    def tree_based_feature_importance(self, n_estimators=100):
        """
        Perform feature selection based on tree-based feature importance.
        
        Args:
        - df: DataFrame containing predictor and target variables
        - target_column: Name of the target variable column
        - n_estimators: Number of trees in the ensemble
        
        Returns:
        - List of selected feature names based on feature importance
        """
        X = self.df.drop(columns=[self.dep_var])
        y = self.df[self.dep_var]
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
        selected_features = list(X.columns[feature_importance > 0])
        return selected_features


    def rfa_feature_selection(self, estimator=RandomForestRegressor()):
        """
        Perform feature selection using Recursive Feature Addition (RFA).
        
        Args:
        - df: DataFrame containing predictor and target variables
        - target_column: Name of the target variable column
        - estimator: Machine learning model or estimator
        
        Returns:
        - List of selected feature names based on RFA
        """
        X = self.df.drop(columns=[self.dep_var])
        y = self.df[self.dep_var]
        rfecv = RFECV(estimator, cv=5)
        rfecv.fit(X, y)
        selected_features = list(X.columns[rfecv.support_])
        return selected_features

   
