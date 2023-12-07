import pandas as pd
import numpy as np 
import QuantLib as ql
from pyfi.core.underlying import Underlying

from datetime import datetime
today = datetime.today()
from enum import Enum

class OptionType(Enum):
    CALL = ql.Option.Call
    PUT = ql.Option.Put

class OptionExposure(Enum):
    LONG = 'Ask'
    SHORT = 'Bid'

RISK_FREE_RATE = 0.05

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ OptionContract                                                                                                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
class Contract(Underlying):

    def __init__(
        self, 
        ticker:str = None,
        option_type:OptionType=None, 
        option_exposure:OptionExposure=None, 
        valuation:ql.Date=None,
        expiration:ql.Date=None,
        premium=None,
        spot=None,
        K=None,
        ivol=None,
        contract_id=None,
        **kwargs
    ):
        """ 
        BSM Options pricing model. Assumes American exercise styles.

        premium: option contract price (credit/debit premium)
        K: strike price
        ivol: market implied volatility
        """
        period = (datetime(expiration.year(), expiration.month(), expiration.dayOfMonth()) - datetime.today()).days

        super().__init__(ticker=ticker, period=period) # spot; hvol; hvol_two_sigma

        if spot is not None:
            self.spot = spot # NOTE: Override inherited self.spot from Underlying()
        else:
            self._spot = self.spot
     
        self.option_type = option_type
        self.option_exposure = option_exposure
        self.valuation = valuation
        self.expiration = expiration
        self.premium = premium
        self.K = K
        self.ivol = ivol 
        self.contract_id = contract_id
        self.rfr = RISK_FREE_RATE

        print(self)


    def __str__(self):
        return str({k: v for k, v in vars(self).items() if not isinstance(v, pd.Series)
                    and not isinstance(v, pd.DataFrame)})


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

        try:
            implied_volatility = option.impliedVolatility(option_price, process)
        except RuntimeError:
            implied_volatility = np.nan

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

        try:
            npv = option.NPV() # RuntimeError: negative probability
        except RuntimeError:
            npv = np.nan

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

        try:
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta()
        except:
            delta, gamma, theta = np.nan, np.nan, np.nan

        return {'delta': delta, 'gamma': gamma, 'theta': theta}



    def build_analysis_frame(self):
        npv_market_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.ivol, 
            self.rfr
        )
        
        npv_historical_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.hvol, 
            self.rfr
        )

        npv_2std_historical_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.hvol_two_sigma, 
            self.rfr
        )

        npv_garch_iv = self.solve_for_npv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.hvol_garch, 
            self.rfr
        )

        iv = self.solve_for_iv(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.premium, 
            self.rfr
        )

        greeks = self.calculate_option_greeks(
            self.option_type, 
            self._spot, 
            self.K, 
            self.expiration, 
            self.valuation, 
            self.ivol, 
            self.rfr
        )
            
        instance_vars =  {key: value for key, value in vars(self).items() if not isinstance(value, pd.DataFrame)}

        res_dict = {
            **instance_vars,
            'npv_market': npv_market_iv, 
            'npv_historical': npv_historical_iv, 
            'hvol2':self.hvol_two_sigma,
            'npv_historical_2std': npv_2std_historical_iv,
            'hvol_garch':self.hvol_garch,
            'npv_garch_iv':npv_garch_iv, 
            'calc_iv': iv, 
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
                 ticker=None,
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

        self.ticker=ticker

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.process_chain()


    def get_params(self, contract:pd.DataFrame):
        
        self.contract_id = contract['Contract Name']
        
        self.params = contract.to_frame().T.set_index(['Contract Name']).to_dict(orient = 'index').get(self.contract_id)
        
        return self.params


    def set_instance_vars(self, params):

        if not hasattr(self, 'spot'):
            self.spot = None

        if not hasattr(self, 'valuation_date'):
            self.valuation_date = ql.Date(today.day, today.month,  today.year) 

        if self.option_exposure == OptionExposure.LONG:
            self.premium = params['Ask']

        elif self.option_exposure == OptionExposure.SHORT:
            self.premium = params['Bid']


    def instantiate_contract(self, data):
            
        params = self.get_params(data)
        # print(params)

        self.set_instance_vars(params)

        option_contract = Contract(
            ticker=self.ticker,
            option_type=self.option_type,
            option_exposure=self.option_exposure,
            valuation=ql.Date(today.day, today.month, today.year),
            expiration=ql.Date(params['Expiration_dt'].day, params['Expiration_dt'].month, params['Expiration_dt'].year),
            premium=self.premium,
            spot=self.spot,
            K=params['Strike'],
            ivol=params['Market_IV'],
            contract_id=self.contract_id
        )

        return option_contract
    

    def process_chain(self):

        frames = []

        for row in self.chain.iterrows():

            ix, data = row
    
            res = self.instantiate_contract(data).build_analysis_frame()

            self.full_res = pd.concat([data, res.T], axis=0)

            self.full_res.columns = self.full_res.loc['Contract Name']

            self.full_res = self.full_res.iloc[1:]

            self.full_res.columns.name = None

            frames.append(self.full_res)
        
        self.processed_chain = pd.concat(frames, axis=1)#.T



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Option Strategies                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

class Strategy:

    def __init__(self) -> None:
        """ A class to be inherited by each option strategy.

        sT: End Price; Series of underlying stock price at time T.
        K: Strike price
        p: Premium (Credit/Debit from purchase)
        """
        pass

    @staticmethod
    def put_payoff(sT, K, p):
        return np.where(sT < K, K - sT, 0) - p

    @staticmethod
    def call_payoff(sT, K, p):
        return np.where(sT > K, sT - K, 0) - p

    @staticmethod
    def get_sT(K):
        return np.arange(K*0.5,K*1.5,1)

    @staticmethod
    def get_pl_ratio(p, l):
        return p / l
    
