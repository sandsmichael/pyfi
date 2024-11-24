import sys, os
import pandas as pd
import requests
import json
from fredapi import Fred
from pathlib import Path
parent = Path(__file__).resolve().parent

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Fred API Key                                                                                                     │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
SECRET_FP = os.path.join(parent, "secrets.json")
with open(SECRET_FP, 'r') as file:
    secrets = json.load(file)
API_KEY = secrets['fred_api_key']

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Standard Features                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
IDS_STD_MACRO = ["GDP", "CPIAUCSL", "UNRATE"]
IDS_STD_FI = ["BAMLH0A0HYM2"]
IDS_STD_FUNDAMENTAL = []
IDS_STD_EQTY = ['SP500']
IDS_STANDARD = [*IDS_STD_MACRO, *IDS_STD_FI, *IDS_STD_FUNDAMENTAL, *IDS_STD_EQTY]

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Request JSON Response from FRED                                                                                  │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def get_matrix(ids = ["GDP", "CPIAUCSL", "UNRATE"]):
    """ Retrieve multiple Fred data series' and return a dataframe. """
    
    merged_data = None
    for series_id in ids:
        params = {
            "series_id": series_id,
            "api_key": API_KEY,
            "file_type": "json",  
        }
        FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
        response = requests.get(FRED_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            observations = data["observations"]
            df = pd.DataFrame(observations)
            df["value"] = pd.to_numeric(df["value"], errors="coerce") 
            df = df[["date", "value"]]  
            df.rename(columns={"value": series_id}, inplace=True) 
            df.set_index("date", inplace=True) 
            
            if merged_data is None:
                merged_data = df 
            
            else:
                merged_data = merged_data.merge(
                    df, how="outer", left_index=True, right_index=True
                ) 
        
        else:
            print(f"Failed to fetch data for {series_id}. Status code: {response.status_code}")
            pass

    merged_data.index = pd.to_datetime(merged_data.index)

    return merged_data


def get_fred_data(series_list, start_date=None, end_date=None):
    fred = Fred(api_key=API_KEY)
    df = pd.DataFrame()
    for series_id in series_list:
        series_data = fred.get_series(series_id, start_date, end_date)
        series_df = pd.DataFrame({series_id: series_data})
        if df.empty:
            df = series_df
        else:
            df = pd.merge(df, series_df, left_index=True, right_index=True, how='outer')
    return df


