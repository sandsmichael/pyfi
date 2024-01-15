from pyfi.core.options.options import Strategy, Contract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class BearCallCreditSpread(Strategy):

    def __init__(
        self,
        n_contracts:int=1,
        clsLong:Contract = None,
        clsShort:Contract = None
    ) -> None:
        
        super().__init__()

        self.n_contracts = n_contracts * 100

        self.sT = self.get_sT(clsShort.K)

        self.short_call_payoff = self.call_payoff(sT=self.sT, K=clsShort.K, p=clsShort.premium) * -1 * self.n_contracts

        self.long_call_payoff = self.call_payoff(sT=self.sT, K=clsLong.K, p=clsLong.premium) * self.n_contracts

        spot_prices_df = pd.DataFrame({'spot': pd.Series(self.sT)})
        
        self.spread_payoff = spot_prices_df.apply(lambda row: self.bear_call_spread_payoff(
            row['spot'], clsLong.K, clsShort.K, clsLong.premium, clsShort.premium), axis=1) * self.n_contracts
        
        self.clsLong = clsLong
        self.clsShort = clsShort

    @property
    def max_profit(self):
        return (self.clsShort.premium - self.clsLong.premium) * self.n_contracts

    @property
    def max_loss(self):
        return ((self.clsShort.K - self.clsLong.K) + (self.clsShort.premium - self.clsLong.premium)) * self.n_contracts

    @property
    def pl_ratio(self):
        return self.get_pl_ratio(self.max_profit, self.max_loss)

    @property
    def breakeven(self):
        short_call_strike = self.clsShort.K
        net_credit = self.clsShort.premium - self.clsLong.premium
        breakeven_point = short_call_strike + net_credit
        return breakeven_point

    def bear_call_spread_payoff(self, spot_price, call_strike_long, call_strike_short, long_call_cost, short_call_credit):
        long_call_payoff = max(spot_price - call_strike_long, 0) - long_call_cost

        short_call_payoff = min(call_strike_short - spot_price, short_call_credit)

        bear_call_spread_payoff = long_call_payoff + short_call_payoff

        return bear_call_spread_payoff

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.sT, self.short_call_payoff, '--', color='r', label='Short Call Payoff', alpha=0.3)
        ax.plot(self.sT, self.long_call_payoff, '--', color='g', label='Long Call Payoff', alpha=0.3)
        ax.plot(self.sT, self.spread_payoff, color='black', label='Spread Payoff')

        ax.scatter(self.sT.min(), self.max_profit, color='lightgreen', s=20, marker='o')
        ax.text(self.sT.min(), self.max_profit, f'Max Profit\n{self.max_profit}', ha='left', va='center', fontsize=8)

        ax.scatter(self.sT.max(), self.max_loss, color='red', s=20, marker='o')
        ax.text(self.sT.max(), self.max_loss, f'Max Loss\n{self.max_loss}', ha='right', va='center', fontsize=8)

        ax.axvline(self.breakeven, color='black', linestyle='--', alpha=0.5)
        ax.text(self.breakeven, self.long_call_payoff.max(), f'Breakeven\n{self.breakeven}', ha='left', va='center', fontsize=8)

        ax.axvline(self.clsShort.spot, color='grey', linestyle='--', alpha=0.5)
        ax.text(self.clsShort.spot, self.short_call_payoff.min(), f'Spot\n{round(self.clsShort.spot, 2)}', ha='left', va='center', fontsize=8)

        ax.grid(axis='x', which='major', visible=False)
        ax.grid(axis='y', which='major', visible=False)

        plt.legend()
        plt.xlabel('Stock Price')
        plt.ylabel('Profit & Loss')
        plt.show()
