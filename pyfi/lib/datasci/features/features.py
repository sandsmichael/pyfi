
import sys, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import re
import itertools
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from more_itertools import distinct_combinations as idc
from more_itertools import distinct_permutations as idp
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

def get_permutations(self):
    return pd.DataFrame(idp(self.df.columns, 2), columns=['ts1', 'ts2'])

def get_combinations(self):
    return pd.DataFrame(idc(self.df.columns, 2), columns=['ts1', 'ts2'])


class Engine:
    pass


class Features():
    """ Functionality for cleaning and pre-processing raw features (not specific to time series data)
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


    def winsorize_columns(self, df, subset = None, limits=[0.05, 0.05]):
        winsorized_df = df.copy()

        if subset is None:
            subset = winsorized_df.columns

        for column in subset:
            winsorized_df[column] = winsorize(df[column], limits=limits)

        return winsorized_df
    

    def standardize(self):
        """ Z-score
        """
        return self.df.apply(lambda x: (x - x.mean()) / x.std())


    def standard_scaler(self, df):
        """ Standardize each series of a dataframe by (observation - mean) /  standard deviation 
        """
        self.scaler = StandardScaler()
        self.scaled_values = self.scaler.fit_transform(df)
        return pd.DataFrame(self.scaled_values, columns=df.columns)
        

    def inverse_standard_scaler(self, df):
        unscaled_values = self.scaler.inverse_transform(df)
        return pd.DataFrame(unscaled_values, columns=df.columns)


    def normalize(self):
        pass

    def de_normalize(self):
        pass

    def binary_encode(self, var_list:str) -> None:
        def binary_map(x):
            return x.map({'yes': 1, "no": 0})
        self.df[var_list] = self.df[var_list].apply(binary_map)


    def cat_encode(self, var_list:str) -> None:
        for var in var_list:
            dummies = pd.get_dummies(self.df[var])
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(var, axis=1, inplace=True)


    def get_feature_ratios(self):

        """
        Calculate the pairwise ratios as a time series for each combination of features in the dataset.

        Parameters:
        data (pd.DataFrame): Time series dataset where each column is a feature.

        Returns:
        pd.DataFrame: A DataFrame where each column is the price ratio of a pair of features.
        """
        data = self.df
        feature_pairs = list(itertools.combinations(data.columns, 2))
        
        ratios = {}
        
        for feature1, feature2 in feature_pairs:
            ratio_series = data[feature1] / data[feature2]
            ratio_name = f"{feature1}/{feature2}" 
            ratios[ratio_name] = ratio_series
        
        feat_ratios_df = pd.DataFrame(ratios, index=data.index)

        z_scores = (feat_ratios_df - feat_ratios_df.mean()) / feat_ratios_df.std()

        return feat_ratios_df, z_scores


    def naive_transform(self, skew_threshold=1, var_threshold=(0.01, 1), normalize_threshold=10):
        """
        Analyze a DataFrame, recommend and apply transformations iteratively for each column,
        and update column names to reflect transformations.

        Parameters:
            df (pd.DataFrame): The input DataFrame to analyze.
            skew_threshold (float): Threshold above which skewness suggests log transformation.
            var_threshold (tuple): A tuple of (low, high) variance thresholds for scaling recommendation.
            normalize_threshold (float): Threshold for the range of values to decide on normalization.

        Returns:
            pd.DataFrame: The transformed DataFrame with updated column names.
            dict: A dictionary with column names as keys and a list of applied transformations as values.
        """
        transformed_df = self.df
        transformations_applied = {}

        for col in transformed_df.columns:
            transformations = []
            feature = transformed_df[col]
            
            # Skip non-numeric columns
            if not np.issubdtype(feature.dtype, np.number):
                continue

            while True:
                applied = False  # Track if a transformation was applied in this iteration

                if "log" not in transformations and skew(feature.dropna()) > skew_threshold and feature.min() > 0:
                    transformed_df[col] = np.log(feature + 1)  # Add 1 to handle zeros safely
                    transformations.append("log")
                    applied = True
                    feature = transformed_df[col]

                var = feature.var()
                if not applied and "scale" not in transformations and (var < var_threshold[0] or var > var_threshold[1]):
                    scaler = StandardScaler()
                    transformed_df[col] = scaler.fit_transform(feature.values.reshape(-1, 1)).flatten()
                    transformations.append("scale")
                    applied = True
                    feature = transformed_df[col]

                if not applied and "normalize" not in transformations and feature.max() - feature.min() > normalize_threshold:
                    normalizer = MinMaxScaler()
                    transformed_df[col] = normalizer.fit_transform(feature.values.reshape(-1, 1)).flatten()
                    transformations.append("normalize")
                    applied = True
                    feature = transformed_df[col]

                if not applied:
                    break

            # Update column name to reflect transformations
            if transformations:
                new_col_name = f"{col}_{'_'.join(transformations)}"
                transformed_df.rename(columns={col: new_col_name}, inplace=True)
                transformations_applied[new_col_name] = transformations

        return transformed_df, transformations_applied


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
        






# def scenario_probabilites(self, n=21, bounds=[-5,5]):
#     """ Probability of an observation more extreme than a specified value.
#     """
#     values = np.linspace(bounds[0], bounds[1], n)
    
#     values = pd.Series([ *values[values != 0], 1e-10, -1e-10]).sort_values(ascending=True)
    
#     def _cdf(s):
#         if s > 0:
#             return 1 - stats.norm(self.mean, self.std_dev).cdf(s)
#         else:
#             return stats.norm(self.mean, self.std_dev).cdf(s)

#     res = values.apply(_cdf)
    
#     if isinstance(self.df, pd.DataFrame):
#         res = pd.DataFrame(res.tolist(), columns = self.df.columns, index = values.round(2)).round(4).T
#     else:
#         res = pd.DataFrame(res.tolist(), columns = [self.df.name], index = values.round(2)).round(4).T

#     res.columns.name = 'P(x)'

#     return res


# def scenario_z_scores(self, n=21, bounds=[-5,5]):
#     """ Z-score for each scenario observation based on corresponding column statistics
#     """
#     values = np.linspace(bounds[0], bounds[1], n)
    
#     values = pd.Series([ *values[values != 0], 1e-10, -1e-10]).sort_values(ascending=True)

#     z_scores = [(value - self.mean) / self.std_dev for value in values]
    
#     res = pd.DataFrame(z_scores, index = values.round(2)).round(4).T

#     res.columns.name = 'Z-score'

#     return res
    

# def get_qq(self):

#     # Example data
#     data = np.random.normal(loc=0, scale=1, size=100)

#     # Get Q-Q plot data
#     qq_data = stats.probplot(data, dist="norm")  # "norm" is the standard normal distribution

#     # Extract quantiles
#     theoretical_quantiles = qq_data[0][0]
#     sample_quantiles = qq_data[0][1]
#     least_squares_fit = qq_data[1]  # slope, intercept, r

#     print("Theoretical Quantiles:", theoretical_quantiles)
#     print("Sample Quantiles:", sample_quantiles)
#     print("Least Squares Fit (slope, intercept, r):", least_squares_fit)


#     def display_missing(self, plot=False):
#         '''
#         Display missing values as a pandas dataframe.
#         '''
#         df = self.df.isna().sum()
#         df = df.reset_index()
#         df.columns = ['features', 'missing_counts']

#         missing_percent = round((df['missing_counts'] / self.df.shape[0]) * 100, 1)
#         df['missing_percent'] = missing_percent

#         if plot:
#             sns.heatmap(self.df.isnull(), cbar=True)
#             plt.show()
#             return df
#         else:
#             return df


#     def detect_outliers(data, n, features):
#         '''
#             Detect Rows with outliers.
#             Parameters
#             '''
#         outlier_indices = []

#         # iterate over features(columns)
#         for col in features:
#             # 1st quartile (25%)
#             Q1 = np.percentile(data[col], 25)
#             # 3rd quartile (75%)
#             Q3 = np.percentile(data[col], 75)
#             # Interquartile range (IQR)
#             IQR = Q3 - Q1

#             # outlier step
#             outlier_step = 1.5 * IQR

#             # Determine a list of indices of outliers for feature col
#             outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index

#             # append the found outlier indices for col to the list of outlier indices
#             outlier_indices.extend(outlier_list_col)

#         # select observations containing more than 2 outliers
#         outlier_indices = Counter(outlier_indices)
#         multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

#         return multiple_outliers




class Generator(Features):
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
    


class Selector(Features):
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

   
