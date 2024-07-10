import pandas as pd
from fredapi import Fred
import json

secret_fp = r'C:\Users\micha\OneDrive\Documents\code\pyfi\pyfi\base\retrievers\secrets.json'
with open(secret_fp, 'r') as file:
    data = json.load(file)


def get_fred_data(series_list, start_date=None, end_date=None):

    fred = Fred(api_key=data['fred_api_key'])

    combined_data = pd.DataFrame()

    for series_id in series_list:

        series_data = fred.get_series(series_id, start_date, end_date)
        
        series_df = pd.DataFrame({series_id: series_data})
        
        if combined_data.empty:
            combined_data = series_df
        else:
            combined_data = pd.merge(combined_data, series_df, left_index=True, right_index=True, how='outer')

    return combined_data


def get_standard_factors():
        # Curve, spreads, swaps, oas, credit, indices, gdp, indprod