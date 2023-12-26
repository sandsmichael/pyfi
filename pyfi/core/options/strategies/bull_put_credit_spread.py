from pyfi.core.options.options import Strategy, Contract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BullPutCreditSpread(Strategy):

    def __init__(
        self,
        n_contracts:int=1,
        clsLong:Contract = None,
        clsShort:Contract = None
    ) -> None:
        
        super().__init__()

        self.n_contracts = n_contracts * 100

        self.sT = self.get_sT(clsLong.K)

        self.long_put_payoff = self.put_payoff(sT = self.sT, K = clsLong.K, p = clsLong.premium)   * self.n_contracts

        self.short_put_payoff = self.put_payoff(sT = self.sT, K = clsShort.K, p = clsShort.premium) * -1  * self.n_contracts

        spot_prices_df = pd.DataFrame({'spot': pd.Series(self.sT)})
        
        self.spread_payoff = spot_prices_df.apply(lambda row: self.bull_put_spread_payoff(
            row['spot'], clsLong.K, clsShort.K, clsLong.premium, clsShort.premium), axis=1)  * self.n_contracts
        
        self.clsLong = clsLong
        self.clsShort = clsShort

    @property
    def max_profit(self):
        return (self.clsShort.premium - self.clsLong.premium) * self.n_contracts

    @property
    def max_loss(self):
        return ((self.clsShort.K - self.clsLong.K) - (self.clsShort.premium - self.clsLong.premium)) *-1 * self.n_contracts

    @property
    def pl_ratio(self):
        return self.get_pl_ratio(self.max_profit, self.max_loss)

    @property
    def breakeven(self):
        short_put_strike = self.clsShort.K
        net_credit = self.clsShort.premium - self.clsLong.premium
        breakeven_point = short_put_strike - net_credit
        return breakeven_point


    def bull_put_spread_payoff(self, spot_price, put_strike_long, put_strike_short, long_put_cost, short_put_credit):
        long_put_payoff = (max(put_strike_long - spot_price, 0) - long_put_cost ) 

        short_put_payoff = min(spot_price - put_strike_short, short_put_credit) 

        bull_put_spread_payoff = long_put_payoff + short_put_payoff

        return bull_put_spread_payoff


    
    def plot(self):
        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(self.sT, self.short_put_payoff,'--', color ='g', label = 'Short Put Payoff', alpha = 0.3)
        ax.plot(self.sT, self.long_put_payoff,'--', color ='r', label ='Long Put Payoff', alpha = 0.3)
        ax.plot(self.sT, self.spread_payoff, color ='black', label ='Spread Payoff')

        ax.axvline(self.clsLong.spot,  color ='lightgrey', linestyle='--', alpha = 0.5)

        ax.scatter(self.sT.max(), self.max_profit, color='lightgreen', s=20, marker='o')
        ax.text(self.sT.max(), self.max_profit, f'Max Profit\n{self.max_profit}', ha='right', va='center',  fontsize=8)

        ax.scatter(self.sT.min(), self.max_loss, color='red', s=20, marker='o')
        ax.text(self.sT.min(), self.max_loss, f'Max Loss\n{self.max_loss}', ha='right', va='center', fontsize=8)

        ax.axvline(self.breakeven, color ='black', linestyle='--', alpha = 0.5)

        ax.grid(axis='x', which='major', visible=False)
        ax.grid(axis='y', which='major', visible=False)

        plt.legend()
        plt.xlabel('Stock Price')
        plt.ylabel('Profit & Loss')
        plt.show()
