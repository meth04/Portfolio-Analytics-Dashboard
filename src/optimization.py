import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Bạn có thể cần cài solver ECOS hoặc SCS:
# pip install ecos scs


def calculate_efficient_frontier(
    prices: pd.DataFrame,
    solver: str = "ECOS",
) -> tuple[dict, dict]:
    """
    Tính Sharpe-maximizing và minimum-volatility portfolio.
    Trả về tuple (weights_sharpe, weights_min_vol).
    """
    # Tính return trung bình và cov
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # Sharpe-maximizing portfolio
    ef_sharpe = EfficientFrontier(mu, S, solver=solver, verbose=False)
    ef_sharpe.max_sharpe()
    weights_sharpe = ef_sharpe.clean_weights()

    # Minimum-volatility portfolio
    ef_min_vol = EfficientFrontier(mu, S, solver=solver, verbose=False)
    ef_min_vol.min_volatility()
    weights_min_vol = ef_min_vol.clean_weights()

    # Vẽ Efficient Frontier với returns_range giới hạn điểm
    ef_plot = EfficientFrontier(mu, S, solver=solver, verbose=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    # Giới hạn số điểm để tránh lỗi solver
    returns_range = np.linspace(mu.min(), mu.max(), 20)
    plotting.plot_efficient_frontier(
        ef_plot,
        ax=ax,
        show_assets=True,
        returns_range=returns_range,
    )
    ax.set_title("Efficient Frontier")
    plt.tight_layout()
    plt.show()

    return weights_sharpe, weights_min_vol


def plot_weights(weights: dict, title: str = "Portfolio Weights"):
    """
    Vẽ biểu đồ phân bổ tỷ trọng danh mục (lọc weight > 1%).
    """
    plt.figure(figsize=(10, 6))
    # chỉ hiện cổ phiếu có weight lớn hơn 1%
    filtered = {k: v for k, v in weights.items() if v > 0.01}
    plt.bar(filtered.keys(), filtered.values())
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Đọc dữ liệu giá
    df = pd.read_csv(
        "../data/processed/sp500_price.csv",
        index_col="Date",
        parse_dates=True,
    )
    df = df.dropna(axis=1)  # loại cổ phiếu thiếu dữ liệu

    # Tính frontier và weights
    weights_sharpe, weights_min_vol = calculate_efficient_frontier(
        df,
        solver="ECOS",  # hoặc "SCS"
    )

    # Vẽ biểu đồ weights
    plot_weights(weights_sharpe, "Tangency Portfolio Weights")
    plot_weights(weights_min_vol, "Minimum Volatility Portfolio Weights")
