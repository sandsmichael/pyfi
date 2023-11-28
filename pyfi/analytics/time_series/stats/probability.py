import pandas as pd
import numpy as np
import scipy.stats as stats

class Probability:

    def __init__(self, df:pd.DataFrame) -> None:
        
        self.df = df
 
        self.mean = self.df.mean()
        
        self.std_dev = self.df.std()

        # print(self.df.tail()) 


    def standardize(self):
        """ Z-score
        """
        return self.df.apply(lambda x: (x - x.mean()) / x.std())


    def scenario_probabilites(self, n=21, bounds=[-5,5]):
        """ Probability of an observation more extreme than a specified value.
        """
        values = np.linspace(bounds[0], bounds[1], n)
        
        values = pd.Series([ *values[values != 0], 1e-10, -1e-10]).sort_values(ascending=True)
        
        def _cdf(s):
            if s > 0:
                return 1 - stats.norm(self.mean, self.std_dev).cdf(s)
            else:
                return stats.norm(self.mean, self.std_dev).cdf(s)

        res = values.apply(_cdf)
        
        res = pd.DataFrame(res.tolist(), columns = self.df.columns, index = values.round(2)).round(4).T

        res.columns.name = 'P(x)'

        return res


    def scenario_z_scores(self, n=21, bounds=[-5,5]):
        """ Z-score for each scenario observation based on corresponding column statistics
        """
        values = np.linspace(bounds[0], bounds[1], n)
        
        values = pd.Series([ *values[values != 0], 1e-10, -1e-10]).sort_values(ascending=True)

        z_scores = [(value - self.mean) / self.std_dev for value in values]
        
        res = pd.DataFrame(z_scores, index = values.round(2)).round(4).T

        res.columns.name = 'Z-score'

        return res