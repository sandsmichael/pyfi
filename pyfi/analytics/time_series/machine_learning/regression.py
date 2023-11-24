import pandas as pd
import numpy as np
from more_itertools import distinct_permutations as idp
from more_itertools import distinct_combinations as idc
from enum import Enum

class RegType(Enum):
    UNIVARIATE = 1
    MULTIVARIATE = 2
    PERMUTATIONS = 3
    COMBINATIONS = 4


class Regression:

    def __init__(self, cls):
        
        self.cls = cls
        
        self.df = self.cls.df
        self.how = self.cls.how

        dep_var = self.cls.dep_var
        indep_var = self.cls.indep_var

        if self.how == RegType.UNIVARIATE:
            self.dep_var = dep_var
            
            if indep_var is not None:
                self.indep_var = indep_var
            else:
                self.indep_var = self.df.columns[1]

        elif self.how == RegType.MULTIVARIATE:
            self.dep_var = dep_var
            
            if indep_var is not None:
                self.indep_var = indep_var
            else:
                self.indep_var = [c for c in self.df.columns if c != dep_var]

        elif self.how == RegType.PERMUTATIONS:
            self.pairs = self.get_permutations()
        
        elif self.how == RegType.COMBINATIONS:
            self.pairs = self.get_combinations()


    def get_combinations(self):
        return pd.DataFrame(idc(self.df.columns, 2), columns=['ts1', 'ts2'])

    def get_permutations(self):
        return pd.DataFrame(idp(self.df.columns, 2), columns=['ts1', 'ts2'])


    def fit(self):

        if not hasattr(self, 'pairs'):
            # fit reg using dep/indeo
            pass
        
        elif hasattr(self, 'pairs'):

            for p in self.pairs:
                pass