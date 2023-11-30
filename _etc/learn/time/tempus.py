import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass, field, asdict

@dataclass
class Tempus:

    data:pd.DataFrame         
    ds:str = None                                   # Datestamp column
    dep_var: str  = None                            # Dependent variable
    indep_var: list = None                          # Independent variable
    ts:pd.DataFrame = field(init=False, repr=False) # Timeseries

    def build_frame(self):
        if self.indep_var is not None:
            self.ts = self.data[[self.dep_var] + self.indep_var]
        else:
            self.indep_var = [c for c in self.data.columns if c != self.dep_var]
            self.ts = self.data

    def check_index(self):
        # print(self.ts.index.__class__)
        if self.ds is None: raise ValueError()
        if isinstance(self.ts.index, pd.core.indexes.range.RangeIndex) and self.ts.index.name is None:
            self.ts.set_index(self.ds, inplace = True)
        elif not isinstance(self.ts.index, pd.DatetimeIndex):
            try:
                self.ts.index = pd.to_datetime(self.ts.index)
            except:
                raise ValueError()

    def __post_init__(self):
        if self.dep_var is None: raise ValueError()
        self.build_frame()
        self.check_index()

    def __repr__(self):
        return f"Tempus {self.dep_var} | {self.indep_var}"



class FirstImpression:

    @classmethod
    def check_stationarity(self, cls:Tempus):
        print(cls)

    @classmethod
    def check_normal(self, cls:Tempus):
        pass

    @classmethod
    def check_na(self, cls:Tempus):
        pass



class FeatureEngine:
    pass



obj = Tempus(data=sns.load_dataset('healthexp'), dep_var='Life_Expectancy', indep_var=None, ds = 'Year')
imp = FirstImpression()
imp.check_stationarity(obj)