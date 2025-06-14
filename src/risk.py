import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
from arch import arch_model
import matplotlib.pyplot as plt


def rolling_vol(daily_returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return daily_returns.rolling(window).std() * np.sqrt(252)


def compute_correlation(daily_returns: pd.DataFrame) -> pd.DataFrame:
    return daily_returns.corr()


def compute_pca(daily_returns: pd.DataFrame, n_components: int = 3):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(daily_returns.fillna(0))
    explained_variance = pca.explained_variance_ratio_
    return components, explained_variance


def value_at_risk(daily_returns: pd.DataFrame, alpha: float = 0.05):
    return daily_returns.quantile(alpha)


def conditional_var(daily_returns: pd.DataFrame, alpha: float = 0.05):
    var = value_at_risk(daily_returns, alpha)
    cvar = daily_returns[daily_returns.le(var)].mean()
    return cvar


def estimate_garch_volatility(daily_returns: pd.DataFrame, symbol: str = 'AAPL', p: int = 1, q: int = 1, horizon: int = 5) -> pd.Series:
    returns = daily_returns[symbol] * 100  # chuyển về %
    returns = returns.dropna()
    
    model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=horizon)
    cond_vol = forecast.variance.iloc[-1] ** 0.5
    return cond_vol


if __name__ == "__main__":
    # Ví dụ sử dụng
    df = pd.read_csv("../data/processed/sp500_price.csv", index_col="Date", parse_dates=True)
    daily_returns = df.pct_change().dropna()

    print("\n Rolling Volatility (30-day):")
    print(rolling_vol(daily_returns).tail())

    print("\n Correlation Matrix:")
    corr = compute_correlation(daily_returns)
    print(corr)
    # plot_correlation_heatmap(corr)

    print("\n PCA Variance Explained:")
    _, ev = compute_pca(daily_returns)
    print(ev)
    # plot_pca_variance(ev)

    print("\n Value at Risk:")
    print(value_at_risk(daily_returns))

    print("\n Conditional VaR:")
    print(conditional_var(daily_returns))

    print("\n Estimate Garch Volatility: ", estimate_garch_volatility(daily_returns, symbol='AAPL'))