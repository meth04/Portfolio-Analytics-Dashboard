import pandas as pd 
import numpy as np  
from typing import List, Optional

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

def treynor_ratio(prices:pd.DataFrame, daily_return_results:pd.DataFrame, beta=1.2, risk_free_rate=0.02):
    return (annualized_return(prices, daily_return_results) - risk_free_rate) / beta

def jensen_alpha(prices:pd.DataFrame, daily_return_results:pd.DataFrame, beta=1.2, market_return=0.1, risk_free_rate=0.02)->pd.DataFrame:
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

def infomation_ratio(
    prices: pd.DataFrame,
    daily_return_results: pd.DataFrame,
    benchmark: pd.Series,
    freq: str = 'daily'
) -> pd.Series:
    """
    Information Ratio: annualized excess return over tracking error.
    """
    bench_ret = benchmark.pct_change().dropna()
    bench_ret = bench_ret.reindex(daily_return_results.index).dropna()
    dr_aligned = daily_return_results.loc[bench_ret.index]

    excess = dr_aligned.sub(bench_ret, axis=0)

    te = excess.std()

    factor = 251 if freq == 'daily' else 12
    ann_excess = excess.mean() * factor
    ann_te = te * np.sqrt(factor)

    return ann_excess / ann_te

def scenario_analysis(prices: pd.DataFrame,
                      weights: Optional[pd.Series] = None,
                      shocks: List[float] = [-0.1, -0.05, 0.0, 0.05, 0.1]) -> pd.Series:
    last = prices.iloc[-1]
    if weights is None:
        weights = pd.Series(1/len(last), index=last.index)
    else:
        weights = weights.reindex(last.index).fillna(0)
    base = (last * weights).sum()
    impacts = {}
    for s in shocks:
        new_val = (last * (1+s) * weights).sum()
        impacts[s] = (new_val/base) - 1
    return pd.Series(impacts)

def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    corr_records = []
    dates = returns.index[window-1:]
    for i, date in enumerate(dates):
        corr_mat = returns.iloc[i:i+window].corr().stack()
        corr_mat.index = pd.MultiIndex.from_product([[date], corr_mat.index.levels[0], corr_mat.index.levels[1]])
        corr_records.append(corr_mat)
    df = pd.concat(corr_records).rename("corr").to_frame()
    return df

def risk_contribution(prices: pd.DataFrame,
                      weights: pd.Series,
                      freq: str = 'daily') -> pd.Series:
    """
    Tính tỉ lệ đóng góp rủi ro cho mỗi tài sản trong danh mục.
    """
    returns = prices.pct_change().dropna()
    cov = returns.cov() * (251 if freq=='daily' else 12)
    w = weights.reindex(cov.index).fillna(0).values
    port_vol = np.sqrt(w @ cov.values @ w)
    mc = cov.values @ w
    # marginal contribution * weight = contribution
    contrib = w * mc / port_vol
    # % of total vol
    pct = contrib / port_vol
    return pd.Series(pct, index=cov.index).sort_values(ascending=False)


if __name__ == "__main__":
    data = pd.read_csv('../data/processed/sp500_price.csv', index_col='Date')
    daily_return_results = daily_return(data)
    results = infomation_ratio(data, daily_return_results, benchmark=data['AAPL'])
    print(results.head(5))