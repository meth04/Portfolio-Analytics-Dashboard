import pandas as pd
import numpy as np
from typing import List, Optional

def daily_return(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how='all')

def cumulative_return(prices: pd.DataFrame, daily_return_results: pd.DataFrame) -> pd.DataFrame:
    cumret = (1 + daily_return_results).cumprod() - 1
    return cumret

def annualized_return(prices: pd.DataFrame, daily_return_results: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    total_days = daily_return_results.shape[0]
    cumulative_returns = cumulative_return(prices, daily_return_results).iloc[-1]
    # annualized return based on trading_days
    results = (1 + cumulative_returns) ** (trading_days / total_days) - 1
    return results

def annualized_volatility(prices: pd.DataFrame, daily_return_results: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    daily_volatility = daily_return_results.std()
    annualized_vol = daily_volatility * np.sqrt(trading_days)
    return annualized_vol

def sharpe_ratio(prices: pd.DataFrame, daily_return_results: pd.DataFrame, risk_free_rate_annual: float = 0.02, trading_days: int = 252) -> pd.Series:
    # Annualized return and vol; risk_free_rate_annual used directly
    ann_ret = annualized_return(prices, daily_return_results, trading_days)
    ann_vol = annualized_volatility(prices, daily_return_results, trading_days)
    sharpe = (ann_ret - risk_free_rate_annual) / ann_vol
    return sharpe

def max_drawdown(prices: pd.DataFrame, daily_return_results: pd.DataFrame) -> pd.Series:
    cumret = cumulative_return(prices, daily_return_results)
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    max_drawdown_results = drawdown.min()
    return max_drawdown_results

def sortino_ratio(prices: pd.DataFrame, daily_return_results: pd.DataFrame, risk_free_rate_annual: float = 0.02, trading_days: int = 252) -> pd.Series:
    # Convert annual risk-free to daily
    rf_daily = (1 + risk_free_rate_annual) ** (1 / trading_days) - 1
    # downside deviation: only returns below rf_daily
    negative_excess = daily_return_results.copy()
    negative_excess = negative_excess[negative_excess < rf_daily]
    downside_deviation = negative_excess.std() * np.sqrt(trading_days)
    ann_ret = annualized_return(prices, daily_return_results, trading_days)
    sortino = (ann_ret - risk_free_rate_annual) / downside_deviation
    return sortino

def calmar_ratio(prices: pd.DataFrame, daily_return_results: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    ann_ret = annualized_return(prices, daily_return_results, trading_days)
    mdd = max_drawdown(prices, daily_return_results)
    return ann_ret / abs(mdd)

def treynor_ratio(prices: pd.DataFrame, daily_return_results: pd.DataFrame, beta: float = 1.2, risk_free_rate_annual: float = 0.02, trading_days: int = 252) -> pd.Series:
    ann_ret = annualized_return(prices, daily_return_results, trading_days)
    return (ann_ret - risk_free_rate_annual) / beta

def jensen_alpha(prices: pd.DataFrame, daily_return_results: pd.DataFrame, beta: float = 1.2, market_return_annual: float = 0.1, risk_free_rate_annual: float = 0.02, trading_days: int = 252) -> pd.Series:
    # expected portfolio return
    expected_portfolio_return = risk_free_rate_annual + beta * (market_return_annual - risk_free_rate_annual)
    ann_ret = annualized_return(prices, daily_return_results, trading_days)
    jensen_alpha_results = ann_ret - expected_portfolio_return
    return jensen_alpha_results

def rolling_volatility(prices: pd.DataFrame, daily_return_results: pd.DataFrame, window: int = 21, trading_days: int = 252) -> pd.DataFrame:
    return daily_return_results.rolling(window=window, min_periods=1).std() * np.sqrt(trading_days)

def value_at_risk(daily_returns: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    # Historical VaR: returns at quantile
    return daily_returns.quantile(alpha)

def conditional_var(daily_returns: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    var = value_at_risk(daily_returns, alpha)
    cvar = daily_returns[daily_returns.le(var)].mean()
    return cvar

def omega_ratio(prices: pd.DataFrame, daily_return_results: pd.DataFrame, risk_free_rate_annual: float = 0.02, trading_days: int = 252) -> pd.Series:
    rf_daily = (1 + risk_free_rate_annual) ** (1 / trading_days) - 1
    excess = daily_return_results - rf_daily
    gain = excess.where(excess > 0).sum(axis=0)
    loss = -excess.where(excess < 0).abs().sum(axis=0)
    omega = gain / loss
    return omega

def information_ratio(
    prices: pd.DataFrame,
    daily_return_results: pd.DataFrame,
    benchmark_returns: pd.Series,
    freq: str = 'daily'
) -> float:
    bench_ret = benchmark_returns.pct_change().dropna()
    bench_ret = bench_ret.reindex(daily_return_results.index).dropna()
    dr_aligned = daily_return_results.loc[bench_ret.index]
    excess = dr_aligned.sub(bench_ret, axis=0)
    te = excess.std()
    factor = 252 if freq == 'daily' else 12
    ann_excess = excess.mean() * factor
    ann_te = te * np.sqrt(factor)
    return ann_excess / ann_te

def scenario_analysis(prices: pd.DataFrame,
                      weights: Optional[pd.Series] = None,
                      shocks: List[float] = [-0.1, -0.05, 0.0, 0.05, 0.1]) -> pd.Series:
    last = prices.iloc[-1]
    if weights is None:
        weights = pd.Series(1 / len(last), index=last.index)
    else:
        weights = weights.reindex(last.index).fillna(0)
    base = (last * weights).sum()
    impacts = {}
    for s in shocks:
        new_val = (last * (1 + s) * weights).sum()
        impacts[s] = (new_val / base) - 1
    return pd.Series(impacts)

def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    # For performance, compute rolling correlation pairwise if needed or average correlation
    return returns.rolling(window).corr()

def risk_contribution(prices: pd.DataFrame,
                      weights: pd.Series,
                      freq: str = 'daily',
                      trading_days: int = 252) -> pd.Series:
    returns = prices.pct_change().dropna()
    cov = returns.cov() * trading_days
    w = weights.reindex(cov.index).fillna(0).values
    port_vol = np.sqrt(w @ cov.values @ w)
    mc = cov.values @ w
    contrib = w * mc / port_vol
    pct = contrib / port_vol
    return pd.Series(pct, index=cov.index).sort_values(ascending=False)
