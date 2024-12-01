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
            # 'entropy': self.data.apply(lambda x: entropy(pd.value_counts(x).values)),
            # 'harmonic_mean': self.data[self.data > 0].apply(lambda x: hmean(x.dropna())),
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

        # stats['is_normal'] = self.data.apply(lambda x: shapiro(x.dropna())[1] > 0.05)

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


class Significance:
    def __init__(self):
        pass


    def annotate_floats_with_stars(self, label_df, pval_df):
            """
            Annotates the coefficient values in the dataframe with stars based on the corresponding p-value matrix.

            Args:
                label_df (pd.DataFrame): DataFrame containing coefficients or values to be labeled with asteriks based on associated p-values.
                pval_df (pd.DataFrame): DataFrame containing p-values.

            Returns:
                pd.DataFrame: DataFrame with coefficients annotated with stars based on p-values.
            """
            
            def add_stars(value, pval):
                """
                Adds significance stars based on the p-value.

                Args:
                    value (float): The coefficient value.
                    pval (float): The corresponding p-value.

                Returns:
                    str: The coefficient value as a string with stars.
                """
                if isinstance(value, float) and isinstance(pval, float):
                    if pval < 0.01:
                        return f"{value:.4f}***"
                    elif pval < 0.05:
                        return f"{value:.4f}**"
                    elif pval < 0.1:
                        return f"{value:.4f}*"
                    else:
                        return f"{value:.4f}"
                return str(value)

            if label_df.shape != pval_df.shape:
                raise ValueError("The coefficient dataframe and p-value dataframe must have the same structure.")

            annotated_coef_df = label_df.copy()

            for col in label_df.columns:
                annotated_coef_df[col] = label_df.apply(
                    lambda row: add_stars(row[col], pval_df.at[row.name, col]), axis=1
                )

            return annotated_coef_df


    def add_significance_levels(self, df):
        """
        Adds three boolean columns to the dataframe indicating whether the 'p-value'
        column is significant at 1%, 5%, and 10% levels.

        Args:
            df (pd.DataFrame): The dataframe with a 'p-value' column.

        Returns:
            pd.DataFrame: The dataframe with added significance columns.
        """

        if 'p_value' not in df.columns:
            raise ValueError("The dataframe must contain a 'p-value' column.")
        
        df['significant_.01'] = df['p_value'] < 0.01
        df['significant_.05'] = df['p_value'] < 0.05
        df['significant_.10'] = df['p_value'] < 0.1
        
        return df