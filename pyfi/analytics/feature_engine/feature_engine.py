from pyfi.core.timeseries import TimeSeries

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

class FeatureEngine(TimeSeries):
    """ feature engineering and selection: Generates feature sets that maximize the explained variance of the dep var.
    """

    def __init__(self, df, dep_var) -> None:
        
        super().__init__(
            df,
            dep_var=None,
        )

        self.curate() # self.df is assigned here

        self.dep_var = dep_var
        

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


    """ 
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │     Feature Generator                                                                                               │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
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

            

    """ 
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │     Feature Selection                                                                                               │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
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

   