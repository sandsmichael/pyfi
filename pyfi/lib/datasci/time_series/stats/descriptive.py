import pandas as pd
import numpy as np
from scipy.stats import shapiro, entropy, hmean
from statsmodels.tsa.stattools import pacf, acf


class Descriptive:
    def __init__(self, data) -> None:
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("Input data must be a pandas DataFrame or Series.")
        
        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            self.data = data.to_frame(name=data.name or "series")
        else:
            self.data = data

    def describe(self):
        stats = {
            'mean': self.data.mean(),
            'median': self.data.median(),
            'std': self.data.std(),
            'var': self.data.var(),
            'skew': self.data.skew(),
            'kurtosis': self.data.kurtosis(),
            'cv': self.data.std() / self.data.mean(),
            'min': self.data.min(),
            'max': self.data.max(),
            'range': self.data.max() - self.data.min(),
            'percentile_range': self.data.quantile(0.9) - self.data.quantile(0.1),
            'missing_count': self.data.isnull().sum(),
            'missing_percentage': self.data.isnull().mean() * 100,
            'unique_count': self.data.nunique(),
            'entropy': self.data.apply(lambda x: entropy(pd.value_counts(x).values)),
            'harmonic_mean': self.data[self.data > 0].apply(lambda x: hmean(x.dropna())),
            'autocorrelation': self.data.apply(lambda x: x.autocorr(lag=1)),
        }

        quantiles = self.data.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles.index = ['q10', 'q25', 'q50', 'q75', 'q90']
        quantiles = quantiles.T

        q1 = self.data.quantile(0.25)
        q3 = self.data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        stats['iqr'] = iqr
        stats['outliers_iqr'] = ((self.data < lower_bound) | (self.data > upper_bound)).sum()

        z_scores = (self.data - self.data.mean()) / self.data.std()
        stats['outliers_zscore'] = (z_scores.abs() > 3).sum()

        stats['is_normal'] = self.data.apply(lambda x: shapiro(x.dropna())[1] > 0.05)

        # Determine lag with highest partial autocorrelation
        def highest_pacf_lag(series):
            try:
                pacf_values = pacf(series.dropna(), nlags=10)
                return np.argmax(np.abs(pacf_values[1:])) + 1  # Skip lag 0 and add 1 for 1-based indexing
            except Exception:
                return np.nan
        stats['highest_pacf_lag'] = self.data.apply(highest_pacf_lag)

        # Determine lag with highest autocorrelation
        def highest_acf_lag(series):
            try:
                acf_values = acf(series.dropna(), nlags=10)
                return np.argmax(np.abs(acf_values[1:])) + 1  # Skip lag 0 and add 1 for 1-based indexing
            except Exception:
                return np.nan
        stats['highest_acf_lag'] = self.data.apply(highest_acf_lag)

        stats_df = pd.DataFrame(stats)
        stats_df = pd.concat([stats_df, quantiles], axis=1)

        return stats_df.T
