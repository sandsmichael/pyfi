import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from dateutil.parser import parse

class Prep:

    def __init__(self) -> None:
        pass

    def curate(self, df):
        return df.dropna(how = 'any', axis=0) # Dropna or fill with mean?

    def winsorize_columns(self, df, subset = None, limits=[0.05, 0.05]):
        winsorized_df = df.copy()

        if subset is None:
            subset = winsorized_df.columns

        for column in subset:
            winsorized_df[column] = winsorize(df[column], limits=limits)

        return winsorized_df
    

    def standard_scaler(self, df):
        """ Standardize each series of a dataframe by (observation - mean) /  standard deviation 
        """
        self.scaler = StandardScaler()
        self.scaled_values = self.scaler.fit_transform(df)
        return pd.DataFrame(self.scaled_values, columns=df.columns)
        

    def inverse_standard_scaler(self, df):
        unscaled_values = self.scaler.inverse_transform(df)
        return pd.DataFrame(unscaled_values, columns=df.columns)


    # def binary_encode(self, var_list:str) -> None:
    #     def binary_map(x):
    #         return x.map({'yes': 1, "no": 0})
    #     self.df[var_list] = self.df[var_list].apply(binary_map)


    # def cat_encode(self, var_list:str) -> None:
    #     for var in var_list:
    #         dummies = pd.get_dummies(self.df[var])
    #         self.df = pd.concat([self.df, dummies], axis=1)
    #         self.df.drop(var, axis=1, inplace=True)


    def convert_dtypes(self, df):
        '''
        Convert datatype of a feature to its original datatype.
        If the datatype of a feature is being represented as a string while the initial datatype is an integer or a float 
        or even a datetime dtype. The convert_dtype() function iterates over the feature(s) in a pandas dataframe and convert the features to their appropriate datatype
        '''
        if df.isnull().any().any() == True:
            raise ValueError("DataFrame contain missing values")
        else:
            i = 0
            changed_dtype = []
            #Function to handle datetime dtype
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


    def get_numeric_cols(self):
        return self.df.select_dtypes(exclude=['object', 'datetime64']).columns


    def cast_numeric_cols(self):
        for c in self.df.columns:
            if c in self.get_numeric_cols():
                self.df[c] = pd.to_numeric(self.df[c])

