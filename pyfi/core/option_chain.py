import pandas as pd
import QuantLib as ql
from datetime import datetime

class OptionChain:

    def __init__(self, chain:pd.DataFrame = None):
        self.chain = chain
    

    def solve_for_iv(self):
        pass


    def solve_for_npv(self):
        contract = self.chain.iloc[27]
        contract_id = contract['Contract Name']
        print(contract_id)
        params = contract.to_frame().T.set_index(['Contract Name']).to_dict(orient = 'index').get(contract_id)
        print(params)

        option_type = ql.Option.Put

        strike_price = params['Strike']
        volatility = params['Market_IV']
        dividend_rate =  params['Dividend']
        spot_price = params['Spot']
        risk_free_rate = params['Rfr']

        maturity_date = ql.Date(params['Expiration_dt'].day, params['Expiration_dt'].month, params['Expiration_dt'].year)
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        today = datetime.today()
        evaluationDate = ql.Date(today.day, today.month,  today.year)
        ql.Settings.instance().evaluationDate = evaluationDate
        
        spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
        )

        dividend_yield = ql.YieldTermStructureHandle(
            ql.FlatForward(evaluationDate, dividend_rate, day_count)
        )

        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(evaluationDate, risk_free_rate, day_count)
        )

        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(evaluationDate, calendar, volatility, day_count)
        )

        bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                                dividend_yield, 
                                                flat_ts, 
                                                flat_vol_ts)
        
        payoff = ql.PlainVanillaPayoff(option_type, strike_price)

        am_exercise = ql.AmericanExercise(evaluationDate, maturity_date)
        american_option = ql.VanillaOption(payoff, am_exercise)

        steps = 200
        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
        american_option.setPricingEngine(binomial_engine)
        print("{:,.10f}".format(american_option.NPV()))