# max drawdowns, stdev, var, bbands
# import talib
import pandas as pd
import numpy as np

def multiindex_from_df_list(cols:list, frames:list):
    """ Accepts a list of pd.DataFrames() in param frames, that have the same column names and structures.
    Builds MultiIndexed columns to group each series of a similarly named column under the level_0 headers passed in cols.

    To retreive an individual grouping from the MultiIndex:
        result.filter(like='ASML', axis=1).droplevel(level=[0], axis=1)

    cols: list of column names to assign
    """
    multi_index = pd.MultiIndex.from_product([cols], names=[None])
    
    result = pd.concat(frames, axis=1, keys=multi_index)
    
    if isinstance(result.columns, pd.MultiIndex) and len(result.columns.levels) > 1:

        result = result.swaplevel(axis=1)

        result = result.sort_index(axis=1, level=[0,1])
    
    return result



class TechnicalAnalysis:
    """ Technical Analysis methods implemented for cross sectional scans
    """

    def __init__(self, df) -> None:
        """ 
        
        """
        self.df = df


    def rsi(self):
        
        change = self.df.diff()
        change.dropna(inplace=True)

        change_up = change.copy()
        change_down = change.copy()

        change_up[change_up<0] = 0
        change_down[change_down>0] = 0

        avg_up = change_up.rolling(14).mean()
        avg_down = change_down.rolling(14).mean().abs()

        rsi = 100 * avg_up / (avg_up + avg_down)

        return rsi


    def bollinger_bands(self):
        sma = self.df.rolling(window=20).mean()
        rstd = self.df.rolling(window=20).std()

        upper_band = sma + 2 * rstd
        lower_band = sma - 2 * rstd

        result = multiindex_from_df_list(cols = ['bb_lower', 'sma', 'bb_upper'], 
                                         frames = [lower_band, sma, upper_band])

        return result


    def max_drawdown(self, window=252):
        
        roll_max = self.df.rolling(window, min_periods=1).max()
        
        dd = self.df / roll_max - 1.0 # underwater 

        max_dd = dd.rolling(window, min_periods=1).min()

        result = multiindex_from_df_list(cols = ['underwater', 'max_dd'], frames = [dd, max_dd])

        return result
    

    def macd(self):
        short_ema = self.df.ewm(span=12, adjust=False).mean()

        long_ema = self.df.ewm(span=26, adjust=False).mean()

        macd_line = short_ema - long_ema

        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        macd_histogram = macd_line - signal_line

        result = multiindex_from_df_list(cols = ['line','signal', 'histogram'], 
                                         frames = [macd_line, signal_line, macd_histogram])
        
        return result
    

    def obv(self):
        data = self.df.copy()
        _df = pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex) and len(data.columns.levels) > 1:

            frames = []
            for stock in data['Close'].columns:
                
                _df['Daily_Return'] = data['Close'][stock].pct_change()
                _df['Direction'] = _df['Daily_Return'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
                _df['Volume_Direction'] = _df['Direction'] * data['Volume'][stock]
                _df['OBV'] = _df['Volume_Direction'].cumsum()
                
                frames.append(_df['OBV'])

            res = multiindex_from_df_list(cols = data['Close'].columns, frames=frames)

        else:
            _df['Daily_Return'] = data['Close'].pct_change()  # Calculate daily returns
            _df['Direction'] = _df['Daily_Return'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            _df['Volume_Direction'] = _df['Direction'] * data['Volume']
            _df['OBV'] = _df['Volume_Direction'].cumsum()

            res = _df['OBV']

        return res