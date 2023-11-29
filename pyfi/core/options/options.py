import pandas as pd
import numpy as np 
import QuantLib as ql

from datetime import datetime
today = datetime.today()
from enum import Enum


class OptionType(Enum):
    CALL = ql.Option.Call
    PUT = ql.Option.Put

class OptionExposure(Enum):
    LONG = 'Ask'
    SHORT = 'Bid'

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ OptionContract                                                                                                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class Contract:

    def __init__(
        self, 
        option_type=None, 
        option_exposure=None, 
        valuation_date=None,
        expiration_date=None,
        underlying_price=None,
        option_price=None,
        strike_price=None,
        risk_free_rate=None,
        market_implied_volatility=None,
        historical_implied_volatility=None,
        historical_2std_implied_volatility=None,
        contract_id=None
    ):
        """ 
        BSM Options pricing model. Assumes American exercise styles.
        """
        self.option_type = option_type
        self.option_exposure = option_exposure
        self.valuation_date = valuation_date
        self.expiration_date = expiration_date
        self.underlying_price = underlying_price
        self.option_price = option_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.market_implied_volatility = market_implied_volatility
        self.historical_implied_volatility = historical_implied_volatility
        self.historical_2std_implied_volatility = historical_2std_implied_volatility  
        self.contract_id = contract_id


    def solve_for_iv(
        self, option_type, underlying_price, strike_price, expiration_date, valuation_date, option_price, risk_free_rate
    ):
        """ 
        https://stackoverflow.com/questions/4891490/calculating-europeanoptionimpliedvolatility-in-quantlib-python
        """
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, strike_price),
            ql.AmericanExercise(valuation_date, expiration_date)
        )
        
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(ql.SimpleQuote(underlying_price)),
            ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, 0, day_count))  # Initial volatility guess (0.2)
        )

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 300)
        option.setPricingEngine(binomial_engine)

        implied_volatility = option.impliedVolatility(option_price, process)

        return implied_volatility


    def solve_for_npv(
        self, option_type, underlying_price, strike_price, expiration_date, valuation_date, implied_volatility, risk_free_rate
    ):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, strike_price), 
            ql.AmericanExercise(valuation_date, expiration_date)
        )
        
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(ql.SimpleQuote(underlying_price)),
            ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, risk_free_rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, calendar, implied_volatility, day_count))
        )  

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", 100)
        option.setPricingEngine(binomial_engine)

        npv = option.NPV()

        return npv


    def calculate_option_greeks(
        self, option_type, underlying_price, strike_price, expiration_date, valuation_date, volatility, risk_free_rate
    ):
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(option_type, strike_price),
            ql.AmericanExercise(valuation_date, expiration_date)
        )
        
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

        return {'delta': delta, 'gamma': gamma, 'theta': theta}



    def build_analysis_frame(self):
        npv_market_iv = self.solve_for_npv(
            self.option_type, 
            self.underlying_price, 
            self.strike_price, 
            self.expiration_date, 
            self.valuation_date, 
            self.market_implied_volatility, 
            self.risk_free_rate
        )
        
        npv_historical_iv = self.solve_for_npv(
            self.option_type, 
            self.underlying_price, 
            self.strike_price, 
            self.expiration_date, 
            self.valuation_date, 
            self.historical_implied_volatility, 
            self.risk_free_rate
        )

        npv_2std_historical_iv = self.solve_for_npv(
            self.option_type, 
            self.underlying_price, 
            self.strike_price, 
            self.expiration_date, 
            self.valuation_date, 
            self.historical_2std_implied_volatility, 
            self.risk_free_rate
        )

        iv = self.solve_for_iv(
            self.option_type, 
            self.underlying_price, 
            self.strike_price, 
            self.expiration_date, 
            self.valuation_date, 
            self.option_price, 
            self.risk_free_rate
        )

        greeks = self.calculate_option_greeks(
            self.option_type, 
            self.underlying_price, 
            self.strike_price, 
            self.expiration_date, 
            self.valuation_date, 
            self.market_implied_volatility, 
            self.risk_free_rate
        )
            
        res_dict = {
            'contract_id': self.contract_id, 
            'underlying_price': self.underlying_price, 
            'valuation_date': self.valuation_date,
            'npv_market': npv_market_iv, 
            'npv_historical': npv_historical_iv, 
            'npv_historical_2std': npv_2std_historical_iv, 
            'iv': iv, 
            **greeks
        }
            
        return pd.DataFrame(res_dict, index=[0])



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ OptionChain                                                                                                      │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class Chain:

    def __init__(self, 
                 chain:pd.DataFrame=None, 
                 option_type = OptionType.CALL, 
                 option_exposure = OptionExposure.LONG, 
                 **kwargs ):
        """ Iterates through an option chain, represented as a pandas dataframe, and instantiates 
        Contract objects to represent each row (contract) on the chain.

        Chain representation is constructed in base.retreivers.options
        
        Returns pd.DataFrame with Contract detail and analytic results
        """
        
        self.chain = chain

        self.option_type = option_type.value
        
        self.option_exposure = option_exposure

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.process_chain()


    def get_params(self, contract:pd.DataFrame):
        
        self.contract_id = contract['Contract Name']
        
        self.params = contract.to_frame().T.set_index(['Contract Name']).to_dict(orient = 'index').get(self.contract_id)
        
        return self.params


    def set_instance_vars(self, params):

        if not hasattr(self, 'underlying_price'):
            self.underlying_price = params['Spot'] 

        if not hasattr(self, 'valuation_date'):
            self.valuation_date = ql.Date(today.day, today.month,  today.year) 

        if self.option_exposure == OptionExposure.LONG:
            self.option_price = params['Ask']

        elif self.option_exposure == OptionExposure.SHORT:
            self.option_price = params['Bid']


    def instantiate_contract(self, data):
            
        params = self.get_params(data)

        self.set_instance_vars(params)

        option_contract = Contract(
            option_type=self.option_type,
            option_exposure=self.option_exposure,
            valuation_date=ql.Date(today.day, today.month, today.year),
            expiration_date=ql.Date(params['Expiration_dt'].day, params['Expiration_dt'].month, params['Expiration_dt'].year),
            underlying_price=self.underlying_price,
            option_price=self.option_price,
            strike_price=params['Strike'],
            risk_free_rate=params['Rfr'],
            market_implied_volatility=params['Market_IV'],
            historical_implied_volatility=params['HistoricalVol'],
            historical_2std_implied_volatility=params['HistoricalVol2Std'],
            contract_id=self.contract_id
        )

        return option_contract
    

    def process_chain(self):

        frames = []

        for row in self.chain.iterrows():

            ix, data = row
    
            try:

                res = self.instantiate_contract(data).build_analysis_frame()

                self.full_res = pd.concat([data, res.T], axis=0)

                self.full_res.columns = self.full_res.loc['Contract Name']

                self.full_res = self.full_res.iloc[1:]

                self.full_res.columns.name = None

                frames.append(self.full_res)
            
            except: pass

        self.processed_chain = pd.concat(frames, axis=1)#.T



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Option Strategies                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

class Strategy:

    def __init__(self) -> None:
        """ A class to be inherited by each option strategy.
        """
        pass


    def put_payoff(self, sT, strike_price, premium):
        return np.where(sT < strike_price, strike_price - sT, 0) - premium




   