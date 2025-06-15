import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, prices: pd.DataFrame, top_n: int = 3, lookback_months: int = 3,
                 rebalance_freq: str = 'M'):
        self.prices = prices
        self.top_n = top_n
        self.lookback_months = lookback_months
        self.rebalance_freq = rebalance_freq
        self.returns = prices.pct_change().dropna()

    def run(self) -> pd.Series:
        portfolio_returns = []
        dates = []
        rebal_dates = self.prices.resample(self.rebalance_freq).last().index
        # Ensure enough periods
        for i in range(self.lookback_months, len(rebal_dates) - 1):
            start = rebal_dates[i - self.lookback_months]
            end = rebal_dates[i]
            next_start = rebal_dates[i]
            next_end = rebal_dates[i + 1]

            past_prices = self.prices.loc[start:end]
            future_returns = self.returns.loc[next_start:next_end]
            if past_prices.empty or future_returns.empty:
                continue
            cum_returns = past_prices.pct_change().add(1).cumprod().iloc[-1] - 1
            top_stocks = cum_returns.sort_values(ascending=False).head(self.top_n).index
            # equal weight
            portfolio_ret = future_returns[top_stocks].mean(axis=1)
            portfolio_returns.append(portfolio_ret)
            dates.append(future_returns.index)
        if portfolio_returns:
            all_returns = pd.concat(portfolio_returns)
            all_returns.index = pd.DatetimeIndex(all_returns.index)
            all_returns.name = "Strategy"
            return all_returns
        else:
            return pd.Series(dtype=float)

def compute_equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()
