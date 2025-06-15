import pandas as pd
import numpy as np
from typing import Optional, Dict
from scipy.stats import norm, t

def historical_var(daily_returns: pd.Series, alpha: float = 0.05) -> float:
    return daily_returns.quantile(alpha)

def historical_cvar(daily_returns: pd.Series, alpha: float = 0.05) -> float:
    var = historical_var(daily_returns, alpha)
    return daily_returns[daily_returns <= var].mean()

def parametric_var_normal(daily_returns: pd.Series, alpha: float = 0.05) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    z = norm.ppf(alpha)
    return mu + sigma * z

def parametric_cvar_normal(daily_returns: pd.Series, alpha: float = 0.05) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    z = norm.ppf(alpha)
    # CVaR for normal: mu - sigma * pdf(z)/alpha
    pdf_z = norm.pdf(z)
    cvar = mu - sigma * pdf_z / alpha
    return cvar

def parametric_var_t(daily_returns: pd.Series, alpha: float = 0.05, df: int = 5) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    # t percent point
    t_ppf = t.ppf(alpha, df)
    # adjust scaling: for Student t, return distribution assumed standardized
    return mu + sigma * t_ppf

def parametric_cvar_t(daily_returns: pd.Series, alpha: float = 0.05, df: int = 5) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    t_ppf = t.ppf(alpha, df)
    # CVaR for t-dist: use formula: mu + sigma * (pdf_t(ppf)/(alpha*(df/(df-1))))
    pdf_t = t.pdf(t_ppf, df)
    # Faktor adjustment: see literature; approximate:
    cvar = mu - sigma * (df + t_ppf**2) / (df - 1) * pdf_t / alpha
    return cvar

def monte_carlo_var(daily_returns: pd.Series, alpha: float = 0.05, num_simulations: int = 10000) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    sims = np.random.normal(mu, sigma, size=num_simulations)
    return np.quantile(sims, alpha)

def monte_carlo_cvar(daily_returns: pd.Series, alpha: float = 0.05, num_simulations: int = 10000) -> float:
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    sims = np.random.normal(mu, sigma, size=num_simulations)
    var = np.quantile(sims, alpha)
    return sims[sims <= var].mean()
