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
IDS_STD_CRYPTO = ['CBBTCUSD']
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



def get_series_info(id):
    """
    Convert a FRED series ID into its associated release ID.

    Parameters:
        series_id (str): The FRED series ID.

    Returns:
        int: The release ID associated with the series, or None if not found.
    """
    FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"
    params = {
        "series_id": id,
        "api_key": API_KEY,
        "file_type": "json",
    }
    response = requests.get(FRED_SERIES_URL, params=params)
    data = pd.DataFrame(response.json()['seriess'])
    return data


def get_all_releases():
    """
    Convert a FRED series ID into its associated release ID.

    Parameters:
        series_id (str): The FRED series ID.

    Returns:
        int: The release ID associated with the series, or None if not found.
    """
    FRED_SERIES_URL = "https://api.stlouisfed.org/fred/releases"
    params = {
        "api_key": API_KEY,
        "file_type": "json",
    }
    response = requests.get(FRED_SERIES_URL, params=params)
    data = pd.DataFrame(response.json()['releases'])
    return data


def get_release_id_from_series_name(id):
    """
    Convert a FRED series ID into its associated release ID.

    Parameters:
        series_id (str): The FRED series ID.

    Returns:
        int: The release ID associated with the series, or None if not found.
    """
    info = get_series_info(id)
    title = info['title']
    releases = get_all_releases()
    rel = releases[releases['name']==title.values[0]]
    return rel


def get_release_dates(rel_id):
    """
    Retrieve the release date for each observation of a given FRED series.

    Parameters:
        series_id (str): The FRED series ID.

    Returns:
        pd.DataFrame: A DataFrame with 'date', 'release_date', and 'value'.
    """
    FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
    release_params = {
        "release_id": rel_id,
        "api_key": API_KEY,
        "file_type": "json",
    }

    response = requests.get(FRED_RELEASE_DATES_URL, params=release_params)
    if response.status_code != 200:
        print(f"Failed to fetch release dates for release_id {rel_id}. Status code: {response.status_code}")
        return pd.DataFrame(columns=["release_date", "series_id"])

    
    data = response.json()
    release_dates = pd.DataFrame(data["release_dates"])
    return release_dates



def get_series_frequency(series_id):
    """
    Fetch the frequency of a FRED series.

    Parameters:
        series_id (str): The FRED series ID.

    Returns:
        str: Frequency of the series (e.g., 'Monthly', 'Daily', 'Quarterly').
    """
    FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
    }

    response = requests.get(FRED_SERIES_URL, params=params)

    if response.status_code == 200:
        series_data = response.json()
        if "seriess" in series_data and len(series_data["seriess"]) > 0:
            frequency = series_data["seriess"][0].get("frequency", "Unknown")
            return frequency
        else:
            print(f"No metadata found for series {series_id}.")
            return "Unknown"
    else:
        print(f"Failed to fetch metadata for {series_id}. Status code: {response.status_code}")
        return "Unknown"






# def get_fred_data(series_list, start_date=None, end_date=None):
#     fred = Fred(api_key=API_KEY)
#     df = pd.DataFrame()
#     for series_id in series_list:
#         series_data = fred.get_series(series_id, start_date, end_date)
#         series_df = pd.DataFrame({series_id: series_data})
#         if df.empty:
#             df = series_df
#         else:
#             df = pd.merge(df, series_df, left_index=True, right_index=True, how='outer')
#     return df


