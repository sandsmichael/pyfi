from pyfi.core.options.options import Strategy, Contract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class VerticalPutSpread(Strategy):

    def __init__(
        self,
        clsLong:Contract = None,
        clsShort:Contract = None
    ) -> None:
        
        super().__init__()

        self.sT = self.get_sT(clsLong.K)

        self.long_put_payoff = self.put_payoff(sT = self.sT, K = clsLong.K, p = clsLong.premium)      

        self.short_put_payoff = self.put_payoff(sT = self.sT, K = clsShort.K, p = clsShort.premium) * -1

        spot_prices_df = pd.DataFrame({'spot': pd.Series(self.sT)})
        
        self.spread_payoff = spot_prices_df.apply(lambda row: self.bear_put_spread_payoff(
            row['spot'], clsLong.K, clsShort.K, clsLong.premium, clsShort.premium), axis=1)


    def bear_put_spread_payoff(self, spot_price, put_sTrike_long, put_sTrike_short, long_put_cosT, short_put_credit):
        long_put_payoff = max(put_sTrike_long - spot_price, 0) - long_put_cosT
        
        short_put_payoff = min(spot_price - put_sTrike_short, short_put_credit)
        
        bear_put_spread_payoff = long_put_payoff + short_put_payoff
        
        return bear_put_spread_payoff

    
    def plot(self):
        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(self.sT, self.short_put_payoff, color ='b', label = '1')
        ax.plot(self.sT, self.long_put_payoff,'--', color ='g', label ='2')
        ax.plot(self.sT, self.spread_payoff,'--', color ='r', label ='3')

        plt.legend()
        plt.xlabel('sTock Price (sT)')
        plt.ylabel('Profit & Loss')
        plt.show()