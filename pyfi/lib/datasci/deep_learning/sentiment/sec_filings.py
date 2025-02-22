import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import time

# SEC API Headers to avoid being blocked
HEADERS = {
    "User-Agent": "YourName your@email.com"
}

# Step 1: Get S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]  # First table contains S&P 500 tickers
    return sp500_df[['Symbol', 'Security']].rename(columns={"Symbol": "Ticker", "Security": "Company"})

# Step 2: Get CIK mappings from SEC
def get_cik_mapping():
    cik_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(cik_url, headers=HEADERS)
    data = response.json()
    
    cik_dict = {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}
    return cik_dict

# Step 3: Get the latest 10-Q filing URL for a given CIK
def get_latest_10q(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        filings = data.get("filings", {}).get("recent", {})
        for form, accession in zip(filings.get("form", []), filings.get("accessionNumber", [])):
            if form == "10-Q":  # Find the latest 10-Q filing
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/index.html"
                return filing_url
    return None

# Step 4: Download the latest 10-Q for all S&P 500 companies
def download_latest_sec_filings():
    sp500_tickers = get_sp500_tickers()
    cik_mapping = get_cik_mapping()

    filings = []
    
    for _, row in sp500_tickers.iterrows():
        ticker, company = row["Ticker"], row["Company"]
        cik = cik_mapping.get(ticker.upper())

        if cik:
            filing_url = get_latest_10q(cik)
            if filing_url:
                filings.append({"Ticker": ticker, "Company": company, "10-Q URL": filing_url})
        
        time.sleep(0.5)  # Respect SEC API rate limits

    return pd.DataFrame(filings)

# Run the function
latest_filings_df = download_latest_sec_filings()
print(latest_filings_df.head())

# Save to CSV
latest_filings_df.to_csv("latest_sp500_10Q_filings.csv", index=False)
