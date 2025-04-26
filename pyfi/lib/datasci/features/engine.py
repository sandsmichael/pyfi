# from pyfi.core.timeseries import TimeSeries

import pandas as pd


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

    The feature engine 1) generates features and 2) selects the best features
    """

    def __init__(self, df, dep_var) -> None:
        
        super().__init__(
            df,
            dep_var=None,
        )

        self.curate() # self.df is assigned here

        self.dep_var = dep_var
        



# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()
# model.fit(X, y)

# importances = model.feature_importances_
# print("Feature Importances:", importances)

