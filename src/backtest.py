import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, prices: pd.DataFrame, top_n: int = 3, lookback_months: int = 3, rebalance_freq: str = 'M'):
        """
        :param prices: DataFrame chứa giá đóng cửa, index là ngày, cột là mã cổ phiếu
        :param top_n: số lượng cổ phiếu chọn mỗi kỳ
        :param lookback_months: số tháng để tính cumulative return
        :param rebalance_freq: tần suất tái cân bằng (theo pandas offset alias, mặc định hàng tháng 'M')
        """
        self.prices = prices
        self.top_n = top_n
        self.lookback_months = lookback_months
        self.rebalance_freq = rebalance_freq
        self.returns = prices.pct_change().dropna()

    def run(self):
        portfolio_returns = []
        dates = []

        rebal_dates = self.prices.resample(self.rebalance_freq).last().index

        for i in range(self.lookback_months, len(rebal_dates) - 1):
            start = rebal_dates[i - self.lookback_months]
            end = rebal_dates[i]
            next_period = (rebal_dates[i], rebal_dates[i + 1])

            past_prices = self.prices.loc[start:end]
            future_returns = self.returns.loc[next_period[0]:next_period[1]]

            cum_returns = past_prices.pct_change().add(1).cumprod().iloc[-1] - 1
            top_stocks = cum_returns.sort_values(ascending=False).head(self.top_n).index

            # trung bình lợi suất các cổ phiếu được chọn
            portfolio_ret = future_returns[top_stocks].mean(axis=1)
            portfolio_returns.append(portfolio_ret)
            dates.append(future_returns.index)

        all_returns = pd.concat(portfolio_returns)
        all_returns.index = pd.DatetimeIndex(all_returns.index)
        all_returns.name = "Momentum"
        return all_returns


def compute_equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def plot_equity_curves(momentum: pd.Series, benchmark: pd.Series):
    equity_mom = compute_equity_curve(momentum)
    equity_bench = compute_equity_curve(benchmark)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_mom, label="Momentum Strategy")
    plt.plot(equity_bench, label="Buy and Hold (Benchmark)")
    plt.legend()
    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load giá đã xử lý
    df = pd.read_csv("../data/processed/sp500_price.csv", index_col="Date", parse_dates=True)

    # Chọn một benchmark (ví dụ AAPL hoặc SPY nếu có)
    benchmark = df['AAPL'].pct_change().dropna()

    # Khởi tạo và chạy chiến lược
    bt = Backtester(prices=df)
    momentum_returns = bt.run()

    # So sánh equity curve
    plot_equity_curves(momentum_returns, benchmark)

    # Xuất kết quả
    summary = pd.DataFrame({
        "Momentum Cumulative Return": compute_equity_curve(momentum_returns).iloc[-1] - 1,
        "Benchmark Cumulative Return": compute_equity_curve(benchmark).iloc[-1] - 1,
    }, index=["Total Return"])

    print("\n📈 Summary:")
    print(summary.T.round(4))
