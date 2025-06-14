# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from metrics import (
    daily_return, cumulative_return, annualized_return, annualized_volatility,
    sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio,
    treynor_ratio, jensen_alpha, rolling_volatility,
    value_at_risk, conditional_var, omega_ratio, infomation_ratio,
    scenario_analysis, rolling_correlation, risk_contribution
)
from backtest import Backtester
from optimization import calculate_efficient_frontier, plot_weights

st.set_page_config(page_title="📈 Portfolio Analytics Dashboard", layout="wide")

RISK_METHODS = [
    "Rolling Volatility", "Scenario Analysis",
    "Risk Contribution", "Stress Testing", "Correlation Dynamics"
]

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.dropna(axis=1)

@st.cache_data
def compute_frontier(prices: pd.DataFrame, solver: str):
    """Return (weights_sharpe, weights_minvol)."""
    return calculate_efficient_frontier(prices, solver)

@st.cache_data
def compute_all(prices: pd.DataFrame, bench: pd.Series = None):
    """Compute returns, cumulative, all metrics, and rolling vol."""
    dr = daily_return(prices)
    cumr = cumulative_return(prices, dr)

    metrics_df = pd.DataFrame({
        "Annual Return": annualized_return(prices, dr),
        "Volatility": annualized_volatility(prices, dr),
        "Sharpe": sharpe_ratio(prices, dr),
        "Max Drawdown": max_drawdown(prices, dr),
        "Sortino": sortino_ratio(prices, dr),
        "Calmar": calmar_ratio(prices, dr),
        "Treynor": treynor_ratio(prices, dr),
        "Jensen Alpha": jensen_alpha(prices, dr),
        "VaR (5%)": value_at_risk(prices, dr),
        "CVaR (5%)": conditional_var(prices, dr),
        "Omega": omega_ratio(prices, dr),
    })

    if bench is not None:
        info = infomation_ratio(prices, dr, bench)
        metrics_df["Information Ratio"] = info

    roll_vol = rolling_volatility(prices, dr)
    return dr, cumr, metrics_df, roll_vol


def main():
    st.title("📈 Portfolio Analytics Dashboard")

    # --- Sidebar configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        df = load_data("../data/processed/sp500_price.csv")
        tickers = df.columns.tolist()

        selected = st.multiselect("Select Tickers", tickers, default=tickers[:5])
        start_date, end_date = st.date_input(
            "Date Range",
            [df.index.min(), df.index.max()]
        )
        include_bench = st.checkbox("Include Benchmark AAPL", True)
        solver = st.selectbox("Optimizer Solver", ["ECOS", "SCS"])
        risk_method = st.selectbox("Risk Analysis Method", RISK_METHODS)
        st.markdown("---")
        st.caption("Made by MeTH04 🚀")

    # --- Data filtering ---
    df = load_data("../data/processed/sp500_price.csv")
    data = df.loc[
        (df.index >= pd.to_datetime(start_date)) &
        (df.index <= pd.to_datetime(end_date)),
        selected
    ]
    bench = None
    if include_bench and "AAPL" in df.columns:
        bench = df["AAPL"].loc[data.index]

    # --- Computations ---
    dr, cumr, metrics_df, roll_vol = compute_all(data, bench)
    w_sharpe, w_minvol = compute_frontier(data, solver)

    # --- Metrics table ---
    st.subheader("📊 Portfolio Metrics")
    fmt = {col: "{:.2%}" for col in metrics_df.columns}
    st.dataframe(metrics_df.style.format(fmt), use_container_width=True)

    # --- Risk Analysis section ---
    st.subheader("📉 Risk Analysis")

    if risk_method == "Rolling Volatility":
        fig = px.line(
            roll_vol,
            labels={"value": "Volatility", "index": "Date"},
            line_shape="spline"
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

    elif risk_method == "Scenario Analysis":
        shocks = st.slider("Shock Range (%)", -20, 20, (-10, 10), step=1)
        step = st.selectbox("Shock Step (%)", [1, 2, 5])
        shock_list = np.arange(shocks[0] / 100, shocks[1] / 100 + 1e-4, step / 100)

        w_mode = st.radio("Use Weights", ["Equal Weight", "Max Sharpe"])
        if w_mode == "Equal Weight":
            weights = pd.Series(1 / len(data.columns), index=data.columns)
        else:
            weights = pd.Series(w_sharpe)

        impacts = scenario_analysis(data, weights=weights, shocks=list(shock_list))
        impacts.index = [f"{int(s*100)}%" for s in impacts.index]

        fig = px.bar(
            x=impacts.index,
            y=impacts.values,
            text=[f"{v:.2%}" for v in impacts.values],
            labels={"x": "Shock", "y": "Portfolio Impact (%)"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    elif risk_method == "Risk Contribution":
        contrib = risk_contribution(data, pd.Series(w_sharpe))
        fig = px.bar(
            contrib.reset_index().rename(columns={0: "Contribution", "index": "Asset"}),
            x="Asset",
            y="Contribution",
            text=[f"{v:.2%}" for v in contrib.values],
            labels={"Contribution": "Risk Contribution (%)"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".2%", title="Risk Contribution")
        st.plotly_chart(fig, use_container_width=True)

    elif risk_method == "Stress Testing":
        stress = scenario_analysis(data, weights=None, shocks=[-0.1, 0.1])
        stress.index = ["-10%", "+10%"]
        fig = px.bar(
            x=stress.index,
            y=stress.values,
            text=[f"{v:.2%}" for v in stress.values],
            labels={"y": "Portfolio Impact (%)"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".2%", title="Stress Testing")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Correlation Dynamics
        window = st.slider("Corr Window (days)", 30, 120, 60)
        rc = rolling_correlation(dr, window)
        avg_corr = rc.groupby(level=0)["corr"].mean()
        fig = px.line(
            avg_corr,
            labels={"value": "Avg Correlation", "index": "Date"},
            line_shape="spline"
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(title="Average Rolling Correlation")
        st.plotly_chart(fig, use_container_width=True)

    # --- Prices & Cumulative Returns ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💵 Prices Over Time")
        fig_price = px.line(
            data,
            labels={"value": "Price", "index": "Date"},
            line_shape="spline"
        )
        fig_price.update_traces(line=dict(width=3))
        st.plotly_chart(fig_price, use_container_width=True)
    with col2:
        st.subheader("📈 Cumulative Returns")
        fig_cum = px.line(
            cumr,
            labels={"value": "Cumulative Return", "index": "Date"},
            line_shape="spline"
        )
        fig_cum.update_traces(line=dict(width=3))
        st.plotly_chart(fig_cum, use_container_width=True)

    # --- Correlation heatmap ---
    st.subheader("🔗 Correlation of Returns")
    corr = dr.corr().round(2)
    fig_corr = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            showscale=True,
        )
    )
    fig_corr.update_layout(
        title="Return Correlation",
        xaxis=dict(ticks="", side="top"),
        yaxis=dict(ticks="", autorange="reversed", scaleanchor="x", scaleratio=1),
        width=600,
        height=600,
        margin=dict(l=100, b=50, t=50, r=50),
    )
    st.plotly_chart(fig_corr, use_container_width=False)

    # --- Backtest Momentum vs Benchmark ---
    st.subheader("🚀 Momentum Strategy vs Benchmark")
    bt = Backtester(data)
    strat_ret = bt.run()
    eq_strat = (1 + strat_ret).cumprod()
    df_eq = pd.DataFrame({"Strategy": eq_strat})
    if bench is not None:
        bench_ret = bench.pct_change().dropna().loc[eq_strat.index]
        df_eq["Benchmark"] = (1 + bench_ret).cumprod()

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=df_eq.index,
            y=df_eq["Strategy"],
            mode="lines",
            name="Strategy",
            line=dict(width=3),
        )
    )
    if "Benchmark" in df_eq:
        fig_bt.add_trace(
            go.Scatter(
                x=df_eq.index,
                y=df_eq["Benchmark"],
                mode="lines",
                name="Benchmark",
                line=dict(width=3),
            )
        )
    fig_bt.update_layout(
        title="Equity Curve: Strategy vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Value",
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # --- Portfolio Optimization side by side ---
    st.subheader("📐 Portfolio Optimization")
    opt1, opt2 = st.columns(2)
    with opt1:
        st.markdown("**Max Sharpe (Tangency) Weights**")
        st.table(pd.Series(w_sharpe).sort_values(ascending=False).to_frame("Weight"))
    with opt2:
        st.markdown("**Min Volatility Weights**")
        st.table(pd.Series(w_minvol).sort_values(ascending=False).to_frame("Weight"))

    st.markdown("---")
    plot_weights(w_sharpe, title="Tangency Portfolio Weights")
    plot_weights(w_minvol, title="Minimum Volatility Portfolio Weights")


if __name__ == "__main__":
    main()
