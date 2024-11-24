import sys
from pathlib import Path
from datetime import datetime, date

current_path = Path(__file__).resolve()
project_root = current_path.parents[2]  
sys.path.append(str(project_root))

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Retrievers                                                                                                       │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def test_equity():
    from pyfi.lib.retrievers import equity
    res = equity.get_historical_data(tickers=['AAPL', 'NVDA'])
    return True if len(res) > 1 else False

def test_fred():
    from pyfi.lib.retrievers import fred
    res = fred.get_matrix(ids = fred.IDS_STANDARD)
    return True if len(res) > 1 else False

def test_options_expirations():
    """ The yahoo fin reader has experienced issues in failing to pull contract expiration dates. Logic defaults to pull next Friday only. """
    from pyfi.lib.retrievers import options
    dates = options.get_expiration_dates(ticker='AAPL')
    print(dates)
    return True if len(dates) > 1 else False

def test_options():
    from pyfi.lib.retrievers import options
    calls, puts = options.get_option_chain('AAPL', date = "2024-11-29", strike_bounds = 0.1)
    print(calls)

def test_options_chain():
    from pyfi.lib.retrievers import options
    calls, puts = options.concat_option_chain('AAPL')
    print(calls)
test_options_chain()



