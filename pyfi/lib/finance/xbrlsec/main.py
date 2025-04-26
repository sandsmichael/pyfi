from xbrl_instance import XbrlInstance
from us_gaap import UsGaap
import json

xbrl = XbrlInstance(fp = './xml/amzn-20220930_htm.xml')
gaap = UsGaap().parse_gaap(xbrli = xbrl)


# xbrl = XbrlInstance( url = "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924_htm.xml" )  #gaap
# xbrl = XbrlInstance( url = "https://www.sec.gov/Archives/edgar/data/842180/000084218021000008/bbva-20201231.xml" )  # ifrs
# xbrl = XbrlInstance(fp = './xml/bbva-20201231.xml')
