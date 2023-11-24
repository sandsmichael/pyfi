import pandas as pd
import QuantLib as ql

from datetime import datetime
today = datetime.today()
from enum import Enum


class OptionType(Enum):
    CALL = 0
    PUT = 1

class OptionExposure(Enum):
    LONG = 0
    SHORT = 1


class OptionChain:

    def __init__(self, chain, opt_type = OptionType.CALL, opt_expo = OptionExposure.LONG, **kwargs ):
        
        print(chain)
        frames = []
        for ix, row in enumerate(chain):
            try:
                opt = OptionContract(contract = chain.iloc[ix], opt_type=opt_type, opt_expo=opt_expo, **kwargs)
                frames.append(opt.full_res)
            except: pass

        self.processed_chain = pd.concat(frames, axis=1).T


class OptionContract:

    def __init__(self, contract:pd.DataFrame = None, opt_type = OptionType.CALL, opt_expo = OptionExposure.LONG, **kwargs):
        """ Assumes American exercise styles.
        Pass BSM inputs thorugh kwargs to override default parameters

        underlying_price : NOTE: Needs to be passed in kwargs as override given no real-time data source.
        """
        self.contract = contract

        self.params = self.get_params(contract)

        if 'underlying_price' not in kwargs:
            self.underlying_price = self.params['Spot'] 
        else:
            self.underlying_price = kwargs['underlying_price'] 

        if 'valuation_date' not in kwargs:
            self.valuation_date = ql.Date(today.day, today.month,  today.year)
        else:
            self.valuation_date = kwargs['valuation_date'] 

        self.strike_price = self.params['Strike']  
        
        self.market_implied_volatility = self.params['Market_IV']  
        
        self.historical_implied_volatility = self.params['HistoricalVol']  

        self.historical_2std_implied_volatility = self.params['HistoricalVol2Std']  
        
        self.risk_free_rate = self.params['Rfr'] 
        
        self.expiration_date = ql.Date(self.params['Expiration_dt'].day, self.params['Expiration_dt'].month, self.params['Expiration_dt'].year)

        if opt_type == OptionType.CALL:
            self.option_type = ql.Option.Call
        
        elif opt_type == OptionType.PUT:
            self.option_type = ql.Option.Put

        if opt_expo == OptionExposure.LONG:
            self.option_price = self.params['Ask']
        
        elif opt_expo == OptionExposure.SHORT:
            self.option_price = self.params['Bid']

        self.process()
      

    def get_params(self, contract):
        self.contract_id = contract['Contract Name']
        self.params = contract.to_frame().T.set_index(['Contract Name']).to_dict(orient = 'index').get(self.contract_id)
        return self.params


    def solve_for_iv(self, option_type, underlying_price, strike_price, expiration_date, valuation_date, option_price, risk_free_rate):
        """ 
        https://stackoverflow.com/questions/4891490/calculating-europeanoptionimpliedvolatility-in-quantlib-python
        """
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(ql.PlainVanillaPayoff(option_type, strike_price), ql.AmericanExercise(valuation_date, expiration_date))
        process = ql.BlackScholesProcess(ql.QuoteHandle(ql.SimpleQuote(underlying_price)),
                                        ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count)),
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, 0, day_count)))  # Initial volatility guess (0.2)

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 300)
        option.setPricingEngine(binomial_engine)

        implied_volatility = option.impliedVolatility(option_price, process)

        return implied_volatility
    

    def solve_for_npv(self,option_type, underlying_price, strike_price, expiration_date, valuation_date, implied_volatility, risk_free_rate):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(ql.PlainVanillaPayoff(option_type, strike_price), ql.AmericanExercise(valuation_date, expiration_date))
        process = ql.BlackScholesProcess(ql.QuoteHandle(ql.SimpleQuote(underlying_price)),
                                        ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count)),
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, implied_volatility, day_count)))  

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 100)
        option.setPricingEngine(binomial_engine)

        npv = option.NPV()

        return npv


    def calculate_option_greeks(self, option_type, underlying_price, strike_price, expiration_date, valuation_date, volatility, risk_free_rate):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(ql.PlainVanillaPayoff(option_type, strike_price), ql.AmericanExercise(valuation_date, expiration_date))
        underlying = ql.SimpleQuote(underlying_price)
        underlying_handle = ql.QuoteHandle(underlying)
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count))
        flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, volatility, day_count))
        process = ql.BlackScholesProcess(underlying_handle, flat_ts, flat_vol_ts)

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 100)
        option.setPricingEngine(binomial_engine)

        delta = option.delta()
        gamma = option.gamma()
        theta = option.theta()
        # vega = option.vega()
        # rho = option.rho()

        return {'delta': delta, 'gamma': gamma, 'theta': theta}


    def process(self):
        npv_market_iv = self.solve_for_npv(self.option_type, 
                                           self.underlying_price, 
                                           self.strike_price, 
                                           self.expiration_date, 
                                           self.valuation_date, 
                                           self.market_implied_volatility, 
                                           self.risk_free_rate)

        npv_historical_iv = self.solve_for_npv(self.option_type, 
                                               self.underlying_price, 
                                               self.strike_price, 
                                               self.expiration_date, 
                                               self.valuation_date, 
                                               self.historical_implied_volatility, 
                                               self.risk_free_rate)

        npv_2std_historical_iv = self.solve_for_npv(self.option_type, 
                                               self.underlying_price, 
                                               self.strike_price, 
                                               self.expiration_date, 
                                               self.valuation_date, 
                                               self.historical_2std_implied_volatility, 
                                               self.risk_free_rate)

        iv = self.solve_for_iv(self.option_type, 
                               self.underlying_price, 
                               self.strike_price, 
                               self.expiration_date, 
                               self.valuation_date, 
                               self.option_price, 
                               self.risk_free_rate)

        greeks = self.calculate_option_greeks(self.option_type, 
                                              self.underlying_price, 
                                              self.strike_price,
                                              self.expiration_date, 
                                              self.valuation_date, 
                                              self.market_implied_volatility, 
                                              self.risk_free_rate)
        
        res_dict = {'contract_id':self.contract_id, 'underlying_price':self.underlying_price, 'valuation_date':self.valuation_date,
                    'npv_market': npv_market_iv, 'npv_historical':npv_historical_iv, 
                    'npv_historical_2std':npv_2std_historical_iv, 'iv': iv, **greeks}
        self.res = pd.DataFrame(res_dict, index=[0])

        self.full_res = pd.concat([self.contract, self.res.T], axis=0)

        self.full_res.columns = self.full_res.loc['Contract Name']

        self.full_res = self.full_res.iloc[1:]

        self.full_res.columns.name = None

        return self.res, self.full_res
