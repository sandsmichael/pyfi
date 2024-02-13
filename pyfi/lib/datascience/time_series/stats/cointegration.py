import pandas as pd
import numpy as np

from more_itertools import distinct_permutations as idp
import statsmodels.tsa.stattools as st
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class Cointegration:

    def __init__(self, df):
        self.df = df

        self.permutations = self.get_permutations()


    def get_permutations(self):
        return pd.DataFrame(idp(self.df.columns, 2), columns=['ts1', 'ts2'])


    @classmethod
    def calculate_cointegration(self, ts1, ts2):
        return st.coint(ts1, ts2, method = 'aeg') # aeg (augmented Engle-Granger)
    

    def run_cointegration(self):
        coint_results = self.permutations.apply(lambda row: 
                                               self.calculate_cointegration( self.df[row['ts1']],
                                                                             self.df[row['ts2']]),
                                                                             axis=1)
        return coint_results
    

    def get_cointegration_summary(self):        
        coint_results = self.run_cointegration()

        summary = self.permutations.copy()

        summary['t_statistic'] = coint_results.apply(lambda x: x[0])
        summary['p_value'] = coint_results.apply(lambda x: x[1])
        
        summary['critical_values_1%'] = coint_results.apply(lambda x: x[2][0])
        summary['critical_values_5%'] = coint_results.apply(lambda x: x[2][1])
        summary['critical_values_10%'] = coint_results.apply(lambda x: x[2][2])
        
        summary['significance_1%'] = summary.apply(lambda row: abs(row['t_statistic']) > abs(row['critical_values_1%']), axis=1)
        summary['significance_5%'] = summary.apply(lambda row: abs(row['t_statistic']) > abs(row['critical_values_5%']), axis=1)
        summary['significance_10%'] = summary.apply(lambda row: abs(row['t_statistic']) > abs(row['critical_values_10%']), axis=1)
        
        summary['significance_p_value'] = summary.apply(lambda row: row['p_value'] < 0.05, axis=1)

        summary['id'] = summary['ts1'].astype(str) + '_' + summary['ts2'].astype(str)

        return summary


    @classmethod
    def calculate_cointegration_johansen(self, data):
        jres = coint_johansen(data, det_order=0, k_ar_diff=0)
        return jres
    

    def run_cointegration_johansen(self):
        coint_results = self.permutations.apply(lambda row: self.calculate_cointegration_johansen( 
                                                                self.df[[row['ts1'],row['ts2']]]
                                                                ), axis=1)
        return coint_results
    

    def get_cointegration_summary_johansen(self):
        """  Summary results of Johansen cointegration test

        
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.JohansenTestResult.html 
        """
        # NOTE: models produce output with multiple arrays for different rank tests. Hardcoded [0] refers to intercept, 1 refers to coef..
        model_array = self.run_cointegration_johansen()

        summary = self.permutations.copy()

        summary['eigenvalue'] = model_array.apply(lambda x: x.eig[0])
        summary['lr1_trace_stat'] = model_array.apply(lambda x: x.lr1[0])
        
        summary['lr1_critical_value_10%'] = model_array.apply(lambda x: x.cvt[1][0]) #NOTE: 0 or 1 in first array
        summary['lr1_critical_value_5%'] = model_array.apply(lambda x: x.cvt[1][1])
        summary['lr1_critical_value_1%'] = model_array.apply(lambda x: x.cvt[1][2])
 
        summary['method'] = model_array.apply(lambda x: x.meth)

        summary['significance_1%'] = summary.apply(lambda row: abs(row['lr1_trace_stat']) > abs(row['lr1_critical_value_1%']), axis=1)
        summary['significance_5%'] = summary.apply(lambda row: abs(row['lr1_trace_stat']) > abs(row['lr1_critical_value_5%']), axis=1)
        summary['significance_10%'] = summary.apply(lambda row: abs(row['lr1_trace_stat']) > abs(row['lr1_critical_value_10%']), axis=1)

        summary['id'] = summary['ts1'].astype(str) + '_' + summary['ts2'].astype(str)

        return summary

      