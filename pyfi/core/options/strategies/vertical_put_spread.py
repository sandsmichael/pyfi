from pyfi.core.options.options import Strategy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class VerticalPutSpread(Strategy):

    def __init__(self) -> None:
        super().__init__()

        self.long_put_payoff = self.put_payoff(sT = np.arange(100,220,1), strike_price = 185.5, premium = -1.1)      # long

        self.short_put_payoff = self.put_payoff(sT = np.arange(100,220,1), strike_price = 185.5, premium = -3.1) * -1 # short

        spot_prices = pd.Series(np.arange(100, 220, 1))
        spot_prices_df = pd.DataFrame({'Spot_Price': spot_prices})
        put_strike_long = 155
        put_strike_short = 188
        long_put_cost = 2
        short_put_credit = 1.5
        spot_prices_df['Spread_Payoff'] = spot_prices_df.apply(lambda row: self.bear_put_spread_payoff(row['Spot_Price'], put_strike_long, put_strike_short, long_put_cost, short_put_credit), axis=1)
        self.spread_payoff = spot_prices_df['Spread_Payoff']


    def bear_put_spread_payoff(self, spot_price, put_strike_long, put_strike_short, long_put_cost, short_put_credit):
        long_put_payoff = max(put_strike_long - spot_price, 0) - long_put_cost
        short_put_payoff = min(spot_price - put_strike_short, short_put_credit)
        bear_put_spread_payoff = long_put_payoff + short_put_payoff
        return bear_put_spread_payoff

    

    def plot(self):
        fig, ax = plt.subplots(figsize=(10,5))
        # ax.spines['bottom'].set_position('zero')
        ax.plot(np.arange(100,220,1), self.short_put_payoff, color ='b', label = '1')
        ax.plot(np.arange(100,220,1), self.long_put_payoff,'--', color ='g', label ='2')
        ax.plot(np.arange(100,220,1), self.spread_payoff,'--', color ='g', label ='3')

        plt.legend()
        plt.xlabel('Stock Price (sT)')
        plt.ylabel('Profit & Loss')
        plt.show()