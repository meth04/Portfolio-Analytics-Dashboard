import pandas as pd 
import numpy as np     

def daily_return(prices: pd.DataFrame)->pd.DataFrame:
    return prices.pct_change().dropna(how='all')

def cumulative_return(prices: pd.DataFrame, daily_return_results:pd.DataFrame)->pd.DataFrame:
    cumret = (1 + daily_return_results).cumprod() - 1
    return cumret

def annualized_return(prices: pd.DataFrame, daily_return_results:pd.DataFrame)->pd.DataFrame:
    total_days = daily_return_results.shape[0]
    cumulative_returns = cumulative_return(prices, daily_return_results).iloc[-1]
    results = (1 + cumulative_returns) ** (251/total_days) - 1
    return results

def annualized_volatility(prices: pd.DataFrame, daily_return_results:pd.DataFrame)->pd.DataFrame:
    daily_volatility = daily_return_results.std()
    annualized_vol = daily_volatility * np.sqrt(251)
    return annualized_vol

def sharpe_ratio(prices: pd.DataFrame, daily_return_results: pd.DataFrame, risk_free_rate=0.02)->pd.DataFrame:
    sharpe_ratio_results = (annualized_return(prices, daily_return_results) - risk_free_rate) / annualized_volatility(prices, daily_return_results)
    return sharpe_ratio_results

def max_drawdown(prices: pd.DataFrame, daily_return_results:pd.DataFrame)->pd.DataFrame:
    cumret = cumulative_return(prices, daily_return_results)
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    max_drawdown_results = drawdown.min()
    return max_drawdown_results

def sortino_ratio(prices: pd.DataFrame, daily_return_results:pd.DataFrame, risk_free_rate=0.02)->pd.DataFrame:
    excess_negative_return = daily_return_results[daily_return_results < risk_free_rate]
    downside_deviation = excess_negative_return.std() * np.sqrt(251)
    sortino_ratio_results = (annualized_return(prices, daily_return_results) - risk_free_rate) / downside_deviation
    return sortino_ratio_results

def calmar_ratio(prices:pd.DataFrame, daily_return_results:pd.DataFrame)->pd.DataFrame:
    return annualized_return(prices, daily_return_results) / abs(max_drawdown(prices, daily_return_results))

def treynor_ratio(prices:pd.DataFrame, beta=1.2, risk_free_rate=0.02):
    return (annualized_return(prices, daily_return_results) - risk_free_rate) / beta

def jensen_alpha(prices:pd.DataFrame, beta=1.2, market_return=0.1, risk_free_rate=0.02)->pd.DataFrame:
    excepted_portfolio_return = risk_free_rate + beta * (market_return - risk_free_rate)
    jensen_alpha_results = annualized_return(prices, daily_return_results) - excepted_portfolio_return
    return jensen_alpha_results

def rolling_volatility(prices:pd.DataFrame, daily_return_results:pd.DataFrame, window:int = 21)->pd.DataFrame:
    return daily_return_results.rolling(window=window, min_periods=1).std() * np.sqrt(251)

def value_at_risk(prices:pd.DataFrame, daily_return_results:pd.DataFrame, alpha:float=0.05)->pd.DataFrame:
    var = daily_return_results.quantile(alpha)
    return var 

def conditional_var(prices:pd.DataFrame, daily_return_results:pd.DataFrame, alpha:float=0.05)->pd.DataFrame:
    var = value_at_risk(prices, daily_return_results, alpha)
    cvar = daily_return_results[daily_return_results.le(var)].mean()
    return cvar

def omega_ratio(prices: pd.DataFrame, daily_return_results:pd.DataFrame, risk_free_rate:float=0.02)->pd.Series:
    excess = daily_return_results - risk_free_rate
    gain = excess.where(excess > 0).sum(axis=0)
    loss = -excess.where(excess < 0).abs().sum(axis=0)
    omega = gain/loss
    return omega

def infomation_ratio(prices:pd.DataFrame, daily_return_results:pd.DataFrame, benchmark:pd.Series, freq='daily')->pd.Series:
    benchmark_returns = benchmark.pct_change().dropna()
    aligned = daily_return_results.align(benchmark_returns, join='inner', axis=0)
    daily_return_results, benchmark_returns = aligned[0], aligned[1]

    excess = daily_return_results.sub(benchmark_returns, axis=0)
    tracking_error = excess.std()
    annual_factor = 251 if freq == 'daily' else 12 
    annualized_excess = excess.mean() * annual_factor
    annualized_tracking_error = tracking_error * np.sqrt(annual_factor)

    info_ratio = annualized_excess / annualized_tracking_error
    return info_ratio

if __name__ == "__main__":
    data = pd.read_csv('../data/processed/sp500_price.csv', index_col='Date')
    daily_return_results = daily_return(data)
    results = infomation_ratio(data, daily_return_results, benchmark=data['AAPL'])
    print(results.head(5))