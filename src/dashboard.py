# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from metrics import (
    daily_return,
    cumulative_return,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    calmar_ratio,
    treynor_ratio,
    jensen_alpha,
    rolling_volatility,
    information_ratio,
    risk_contribution,
)
from risk import (
    historical_var,
    historical_cvar,
    parametric_var_normal,
    parametric_cvar_normal,
    parametric_var_t,
    parametric_cvar_t,
    monte_carlo_var,
    monte_carlo_cvar,
)
from backtest import Backtester
from optimization import calculate_efficient_frontier
from pypfopt import expected_returns, risk_models, EfficientFrontier

st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")

# --- Helper functions ---

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.dropna(axis=1, how="all")
    return df

@st.cache_data
def compute_frontier(prices: pd.DataFrame, solver: str):
    return calculate_efficient_frontier(prices, solver)

def format_pct(df: pd.DataFrame) -> dict:
    return {col: "{:.2%}" for col in df.columns}

# Compute rolling Sharpe for a return series
def compute_rolling_sharpe(returns: pd.Series, window: int, risk_free_rate_annual: float) -> pd.Series:
    rf_daily = (1 + risk_free_rate_annual) ** (1/252) - 1
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate_annual) / rolling_vol
    return rolling_sharpe

# Compute drawdown duration
def compute_drawdown_duration(equity: pd.Series) -> pd.DataFrame:
    high_water = equity.cummax()
    df = pd.DataFrame({'equity': equity, 'high_water': high_water})
    df['drawdown'] = df['equity'] / df['high_water'] - 1
    durations = []
    in_dd = False
    peak = None
    trough = None
    for date in df.index:
        if not in_dd:
            if df.loc[date, 'drawdown'] < 0:
                in_dd = True
                peak = df.loc[:date, 'high_water'].iloc[-1]
                trough = date
        else:
            if df.loc[date, 'equity'] >= peak:
                durations.append({'start': trough, 'end': date, 'duration_days': (date - trough).days})
                in_dd = False
    if durations:
        return pd.DataFrame(durations)
    else:
        return pd.DataFrame(columns=['start','end','duration_days'])

# Monte Carlo simulation of equity curves
def simulate_equity(mu: float, sigma: float, days: int, sims: int, start_value: float=1.0) -> pd.DataFrame:
    mu_daily = mu / 252
    sigma_daily = sigma / np.sqrt(252)
    rng = np.random.default_rng()
    rand = rng.standard_normal((days, sims))
    ret = mu_daily + sigma_daily * rand
    equity = np.cumprod(1 + ret, axis=0) * start_value
    df = pd.DataFrame(equity, index=pd.RangeIndex(start=0, stop=days), columns=[f'Sim{i+1}' for i in range(sims)])
    return df

# Stress test historical scenario performance
def performance_in_period(data: pd.DataFrame, weights: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sub = data.loc[(data.index >= start) & (data.index <= end)]
    if sub.empty:
        return pd.Series(dtype=float)
    dr_sub = daily_return(sub)
    port_ret = dr_sub.dot(weights.reindex(dr_sub.columns).fillna(0))
    return (1 + port_ret).cumprod()

# --- Sidebar configuration ---
st.sidebar.header("⚙️ Configuration")

def main():
    default_path = os.path.join("..", "data", "processed", "sp500_price.csv")
    data_path = st.sidebar.text_input("Path to price CSV", value=default_path, key="data_path")
    try:
        df_all = load_data(data_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load data: {e}")
        st.stop()

    rf_input = st.sidebar.number_input("Annual Risk-free Rate (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1, key="rf_input")
    risk_free_rate_annual = rf_input / 100.0
    total_invest = st.sidebar.number_input("Total Investment Amount", min_value=0.0, value=10000.0, step=1000.0, key="total_invest")

    st.sidebar.subheader("VaR/CVaR Settings")
    var_method = st.sidebar.selectbox("VaR Method", ["Historical", "Parametric Normal", "Parametric t-Distribution", "Monte Carlo"], key="var_method")
    confidence_level = st.sidebar.slider("VaR Confidence Level (%)", min_value=90, max_value=99, value=95, step=1, key="conf_level")
    alpha_var = 1 - (confidence_level / 100.0)
    if var_method == "Parametric t-Distribution":
        df_input = st.sidebar.number_input("Degrees of Freedom for t", min_value=2, max_value=30, value=5, step=1, key="df_input")
    else:
        df_input = None
    if var_method == "Monte Carlo":
        num_sims = st.sidebar.number_input("MC Simulations", min_value=1000, max_value=100000, value=10000, step=1000, key="num_sims")
    else:
        num_sims = None

    st.sidebar.subheader("Backtest Settings")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["W", "M", "Q", "A"], index=1, key="rebalance_freq")
    lookback_months = st.sidebar.number_input("Lookback Period (months)", min_value=1, max_value=24, value=3, step=1, key="lookback_months")
    top_n = st.sidebar.number_input("Number of Stocks in Portfolio", min_value=1, max_value=10, value=3, step=1, key="top_n")

    available_tickers = df_all.columns.tolist()
    benchmark_options = [None] + available_tickers
    default_bench = "AAPL" if "AAPL" in available_tickers else None
    bench_index = benchmark_options.index(default_bench) if default_bench else 0
    benchmark_choice = st.sidebar.selectbox("Benchmark Ticker", options=benchmark_options, index=bench_index, key="benchmark_choice")

    st.sidebar.subheader("Per-Stock VaR")
    stock_choice = st.sidebar.selectbox("Select Stock for VaR", options=available_tickers, key="stock_choice")
    investment_amount = st.sidebar.number_input("Investment Amount (USD)", min_value=0.0, value=10000.0, step=1000.0, key="investment_amount")

    st.sidebar.markdown("---")
    st.sidebar.caption("Made by MeTH04")

    st.title("Portfolio Analytics Dashboard")

    min_date = df_all.index.min().date()
    max_date = df_all.index.max().date()
    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date, key="date_range")
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()
    selected = st.multiselect("Select Tickers", options=available_tickers, default=available_tickers[:5], key="selected_tickers")
    if not selected:
        st.error("Select at least one ticker.")
        st.stop()
    data = df_all.loc[(df_all.index.date >= start_date) & (df_all.index.date <= end_date), selected]
    if data.empty:
        st.warning("No data for selected tickers/date range.")
        st.stop()
    bench_series = None
    if benchmark_choice:
        if benchmark_choice in df_all.columns:
            bench_series = df_all[benchmark_choice].loc[data.index]

    dr = daily_return(data)
    cumr = cumulative_return(data, dr)

    # --- Portfolio Metrics ---
    st.subheader("Portfolio Metrics")
    metrics_df = pd.DataFrame({
        "Annual Return": annualized_return(data, dr),
        "Volatility": annualized_volatility(data, dr),
        "Sharpe": sharpe_ratio(data, dr, risk_free_rate_annual),
        "Max Drawdown": max_drawdown(data, dr),
        "Sortino": sortino_ratio(data, dr, risk_free_rate_annual),
        "Calmar": calmar_ratio(data, dr),
        "Treynor": treynor_ratio(data, dr, risk_free_rate_annual),
        "Jensen Alpha": jensen_alpha(data, dr, market_return_annual=0.1, risk_free_rate_annual=risk_free_rate_annual),
    })
    if bench_series is not None:
        bench_ret = bench_series.pct_change().dropna()
        bench_ret = bench_ret.reindex(dr.index).dropna()
        dr_aligned = dr.loc[bench_ret.index]
        try:
            ir = information_ratio(data.loc[dr_aligned.index], dr_aligned, benchmark_returns=bench_series.loc[dr_aligned.index])
            metrics_df["Information Ratio"] = ir
        except Exception:
            pass
    fmt = format_pct(metrics_df)
    st.dataframe(metrics_df.style.format(fmt), use_container_width=True)

    # --- Risk Analysis: Rolling Volatility ---
    st.subheader("Risk Analysis: Rolling Volatility")
    window = st.slider("Rolling Volatility Window (days)", min_value=10, max_value=120, value=21, step=1, key="rv_window")
    roll_vol = rolling_volatility(data, dr, window=window)
    fig_rv = px.line(
        roll_vol,
        labels={"value": "Volatility", "index": "Date"},
        line_shape="spline"
    )
    fig_rv.update_traces(line=dict(width=1))
    st.plotly_chart(fig_rv, use_container_width=True)

    st.subheader("Top 10 stocks carrying the market")
    if bench_series is not None:
        # Compute beta relative to benchmark
        bench_ret = bench_series.pct_change().dropna()
        betas = {}
        for ticker in selected:
            ret = dr[ticker].reindex(bench_ret.index).dropna()
            common_idx = ret.index.intersection(bench_ret.index)
            if len(common_idx) > 1:
                cov = np.cov(ret.loc[common_idx], bench_ret.loc[common_idx])[0,1]
                var_b = np.var(bench_ret.loc[common_idx])
                beta = cov/var_b if var_b>0 else np.nan
                betas[ticker] = beta
        if betas:
            beta_series = pd.Series(betas)
            top10 = beta_series.abs().sort_values(ascending=False).head(10)
            df_beta = pd.DataFrame({"Beta": beta_series}).loc[top10.index]
            st.write("Top 10 stocks by beta vs benchmark:")
            st.dataframe(df_beta.style.format({"Beta": "{:.2f}"}))
            fig_beta = px.bar(df_beta.abs().reset_index().rename(columns={"index":"Ticker", "Beta":"Beta"}), x="Ticker", y="Beta", title="Absolute beta vs. benchmark")
            st.plotly_chart(fig_beta, use_container_width=True)
    else:
        # Equal weight risk contribution
        if selected:
            w_eq = pd.Series(1/len(selected), index=selected)
            rc = risk_contribution(data, w_eq).mul(100)
            top10 = rc.sort_values(ascending=False).head(10)
            df_rc = pd.DataFrame({"Risk Contribution (%)": top10})
            st.write("Top 10 stocks bearing the brunt of risk (equal weight):")
            st.dataframe(df_rc.style.format({"Risk Contribution (%)": "{:.2f}%"}))
            fig_rc_top = px.bar(df_rc.reset_index().rename(columns={"index":"Ticker", "Risk Contribution (%)":"Risk Contribution (%)"}), x="Ticker", y="Risk Contribution (%)", title="Risk Contribution (%) Equal Weight")
            st.plotly_chart(fig_rc_top, use_container_width=True)


    # --- Price & Cumulative Return Plots ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prices Over Time")
        fig_price = px.line(
            data,
            labels={"value": "Price", "index": "Date"},
            line_shape="spline"
        )
        fig_price.update_traces(line=dict(width=1))
        st.plotly_chart(fig_price, use_container_width=True)
    with col2:
        st.subheader("Cumulative Returns")
        fig_cum = px.line(
            cumr,
            labels={"value": "Cumulative Return", "index": "Date"},
            line_shape="spline"
        )
        fig_cum.update_traces(line=dict(width=1))
        st.plotly_chart(fig_cum, use_container_width=True)

    # --- Drawdown Chart ---
    st.subheader("Portfolio Drawdown")
    portfolio_ret = dr.mean(axis=1)
    portfolio_equity = (1 + portfolio_ret).cumprod()
    portfolio_high = portfolio_equity.cummax()
    portfolio_drawdown = portfolio_equity / portfolio_high - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=portfolio_drawdown.index, y=portfolio_drawdown, fill='tozeroy', name='Drawdown', line=dict(color='red')))
    fig_dd.update_layout(title="Portfolio Drawdown Over Time", xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%")
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Correlation Heatmap ---
    st.subheader("Return Correlation Heatmap")
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

    # --- Custom Portfolio Weights ---
    st.subheader("Custom Portfolio Weights")
    raw_weights = {}
    for ticker in selected:
        raw_weights[ticker] = st.slider(f"Weight for {ticker}", min_value=0.0, max_value=1.0, value=1.0/len(selected), step=0.01, key=f"w_{ticker}")
    raw_w = pd.Series(raw_weights)
    if raw_w.sum() > 0:
        weights_custom = raw_w / raw_w.sum()
    else:
        weights_custom = pd.Series(1/len(selected), index=selected)
    st.write("**Normalized Weights:**")
    st.write(weights_custom.apply(lambda x: f"{x:.2%}"))
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    exp_ret_custom = np.dot(weights_custom.values, mu.reindex(selected).values)
    var_custom = np.dot(weights_custom.values, np.dot(S.reindex(index=selected, columns=selected).values, weights_custom.values))
    exp_vol_custom = np.sqrt(var_custom)
    sharpe_custom = (exp_ret_custom - risk_free_rate_annual) / exp_vol_custom if exp_vol_custom>0 else np.nan
    st.write("**Custom Portfolio Expected Metrics:**")
    st.write(f"Expected Annual Return: {exp_ret_custom:.2%}")
    st.write(f"Annual Volatility: {exp_vol_custom:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_custom:.2f}")

    # --- Fixed-weight Partial Optimization ---
    st.subheader("Partial Optimization with Fixed Weight")
    fix_ticker = st.selectbox("Select Ticker to Fix Weight (or None)", options=['AAPL'] + selected, key="fix_ticker")
    if fix_ticker:
        fix_weight = st.slider(f"Fixed Weight for {fix_ticker}", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="fix_weight")
        if fix_weight < 1.0:
            remaining = [t for t in selected if t != fix_ticker]
            if remaining:
                # optimize on remaining to max Sharpe
                mu_rem = mu.reindex(remaining)
                S_rem = S.reindex(index=remaining, columns=remaining)
                try:
                    ef = EfficientFrontier(mu_rem, S_rem)
                    ef.max_sharpe()
                    w_rem = ef.clean_weights()
                    # scale to sum = 1 - fix_weight
                    series_rem = pd.Series(w_rem)
                    series_rem = series_rem / series_rem.sum() * (1 - fix_weight)
                    combined = series_rem.copy()
                    combined[fix_ticker] = fix_weight
                    combined = combined[selected]
                    st.write("**Optimized Weights with Fixed:**")
                    st.write(combined.apply(lambda x: f"{x:.2%}"))
                    # metrics
                    exp_ret_fix = np.dot(combined.values, mu.reindex(selected).values)
                    var_fix = np.dot(combined.values, np.dot(S.reindex(index=selected, columns=selected).values, combined.values))
                    vol_fix = np.sqrt(var_fix)
                    sharpe_fix = (exp_ret_fix - risk_free_rate_annual) / vol_fix if vol_fix>0 else np.nan
                    st.write(f"Expected Annual Return: {exp_ret_fix:.2%}")
                    st.write(f"Annual Volatility: {vol_fix:.2%}")
                    st.write(f"Sharpe Ratio: {sharpe_fix:.2f}")
                except Exception as e:
                    st.warning(f"Partial optimization failed: {e}")
        else:
            st.warning("Fixed weight must be less than 1.")

    # --- Suggested Optimal Allocation ---
    st.subheader("Suggested Optimal Allocation")
    if len(selected) >= 2:
        solver = st.selectbox("Optimizer Solver Allocation", ["ECOS", "SCS"], index=0, key="opt_solver_allocation")
        try:
            w_sharpe, w_minvol = compute_frontier(data, solver)
            df_alloc_pct = pd.DataFrame({'Max Sharpe': pd.Series(w_sharpe), 'Min Volatility': pd.Series(w_minvol)}).loc[selected]
            st.write("**Weights Optimal Portfolios:**")
            st.dataframe(df_alloc_pct.apply(lambda x: x*100).rename(columns=lambda c: f"{c} (%)").style.format("{:.2f}%"))
            results_opt = []
            for label, w_dict in [('Max Sharpe', w_sharpe), ('Min Volatility', w_minvol)]:
                w = pd.Series(w_dict).reindex(selected).fillna(0)
                exp_ret = np.dot(w.values, mu.reindex(selected).values)
                var_p = np.dot(w.values, np.dot(S.reindex(index=selected, columns=selected).values, w.values))
                exp_vol = np.sqrt(var_p)
                sharpe = (exp_ret - risk_free_rate_annual) / exp_vol if exp_vol>0 else np.nan
                results_opt.append({'Portfolio': label,
                                    'Expected Annual Return (%)': exp_ret*100,
                                    'Annual Volatility (%)': exp_vol*100,
                                    'Sharpe Ratio': sharpe})
            df_opt_metrics = pd.DataFrame(results_opt)
            st.write("**Optimal Portfolios Expected Metrics:**")
            st.dataframe(df_opt_metrics.style.format({
                'Expected Annual Return (%)':'{:.2f}%',
                'Annual Volatility (%)':'{:.2f}%',
                'Sharpe Ratio':'{:.2f}'
            }), use_container_width=True)
        except Exception as e:
            st.error(f"Allocation not calculated: {e}")
    else:
        st.info("Choose at least 2 tickers to optimize allocation.")

    # --- Distribution of Returns ---
    st.subheader("Distribution of Daily Returns")
    df_ret_long = dr.melt(var_name='Ticker', value_name='Return').dropna()
    fig_dist = px.histogram(df_ret_long, x='Return', color='Ticker', nbins=50, marginal='box', opacity=0.7)
    fig_dist.update_layout(barmode='overlay', title='Histogram of Daily Returns')
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Rolling Sharpe for custom portfolio ---
    st.subheader("Rolling Sharpe Ratio (Custom Portfolio)")
    window_sharpe = st.slider("Rolling Sharpe Window (days)", min_value=10, max_value=120, value=60, step=5, key="rs_window")
    port_ret_custom = dr.dot(weights_custom.reindex(dr.columns).fillna(0))
    rolling_sharpe = compute_rolling_sharpe(port_ret_custom, window_sharpe, risk_free_rate_annual)
    fig_rs = px.line(x=rolling_sharpe.index, y=rolling_sharpe.values, labels={'x':'Date','y':'Rolling Sharpe'})
    fig_rs.update_layout(title='Rolling Sharpe Ratio')
    st.plotly_chart(fig_rs, use_container_width=True)

    # --- Risk Contribution Bar for optimal portfolios ---
    st.subheader("Risk Contribution (Optimal Portfolios)")
    if len(selected) >= 2:
        try:
            w_sh = pd.Series(w_sharpe).reindex(selected).fillna(0)
            rc_sh = risk_contribution(data, w_sh)
            w_mv = pd.Series(w_minvol).reindex(selected).fillna(0)
            rc_mv = risk_contribution(data, w_mv)
            df_rc = pd.DataFrame({
                'Max Sharpe': rc_sh.mul(100),
                'Min Volatility': rc_mv.mul(100)
            })
            fig_rc = px.bar(df_rc, barmode='group', labels={'value':'Risk Contribution (%)','index':'Ticker'})
            fig_rc.update_layout(title='Risk Contribution per Asset')
            st.plotly_chart(fig_rc, use_container_width=True)
        except Exception as e:
            st.warning(f"Không tính được Risk Contribution: {e}")

    # --- Monte Carlo Simulations for optimal portfolio ---
    st.subheader("Monte Carlo Simulation of Equity Curve (Optimal Portfolio)")
    if len(selected) >= 2:
        sim_horizon = st.number_input("Simulation Horizon (days)", min_value=30, max_value=252*5, value=252, step=30, key="sim_horizon")
        sim_sims = st.number_input("Number of Simulations", min_value=100, max_value=5000, value=500, step=100, key="sim_sims")
        try:
            w_sh = pd.Series(w_sharpe).reindex(selected).fillna(0)
            exp_ret = np.dot(w_sh.values, mu.reindex(selected).values)
            var_p = np.dot(w_sh.values, np.dot(S.reindex(index=selected, columns=selected).values, w_sh.values))
            exp_vol = np.sqrt(var_p)
            df_sim = simulate_equity(exp_ret, exp_vol, days=int(sim_horizon), sims=int(sim_sims), start_value=1)
            fig_mc = go.Figure()
            for col in df_sim.columns:
                fig_mc.add_trace(go.Scatter(x=df_sim.index, y=df_sim[col], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
            fig_mc.update_layout(title='Monte Carlo Equity Simulation (Max Sharpe Portfolio)', xaxis_title='Day', yaxis_title='Equity (normalized)')
            st.plotly_chart(fig_mc, use_container_width=True)
        except Exception as e:
            st.warning(f"Monte Carlo simulation failed: {e}")

    # --- Drawdown Duration ---
    st.subheader("Drawdown Duration (Custom Portfolio)")
    if not port_ret_custom.empty:
        eq_custom = (1 + port_ret_custom).cumprod()
        df_dd = compute_drawdown_duration(eq_custom)
        if not df_dd.empty:
            st.write("Drawdown cycles and recovery times (days):")
            st.dataframe(df_dd)
            fig_dd = px.bar(df_dd, x='start', y='duration_days', labels={'start':'Drawdown Start','duration_days':'Duration (days)'}, title='Drawdown Duration per Cycle')
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.write("No drawdown cycle (never dropped from peak).")

    # --- Backtest Momentum Strategy vs Benchmark ---
    st.subheader("Momentum Strategy Backtest vs Benchmark")
    bt = Backtester(prices=data, top_n=int(top_n), lookback_months=int(lookback_months), rebalance_freq=rebalance_freq)
    strat_returns = bt.run()
    if strat_returns.empty:
        st.warning("No strategy returns computed: check lookback period or data availability.")
    else:
        eq_strat = (1 + strat_returns).cumprod()
        df_eq = pd.DataFrame({"Strategy": eq_strat})
        if benchmark_choice and benchmark_choice in df_all.columns:
            bench_ret_full = df_all[benchmark_choice].pct_change().dropna()
            bench_ret = bench_ret_full.reindex(strat_returns.index).dropna()
            if not bench_ret.empty:
                eq_bench = (1 + bench_ret).cumprod()
                df_eq = df_eq.join(pd.DataFrame({"Benchmark": eq_bench}), how="inner")
        fig_bt = go.Figure()
        fig_bt.add_trace(
            go.Scatter(
                x=df_eq.index,
                y=df_eq["Strategy"],
                mode="lines",
                name="Strategy",
                line=dict(width=1),
            )
        )
        if "Benchmark" in df_eq.columns:
            fig_bt.add_trace(
                go.Scatter(
                    x=df_eq.index,
                    y=df_eq["Benchmark"],
                    mode="lines",
                    name="Benchmark",
                    line=dict(width=1),
                )
            )
        fig_bt.update_layout(
            title="Equity Curve: Strategy vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Value",
            legend=dict(x=0, y=1),
            height=500,
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        total_strat = eq_strat.iloc[-1] - 1
        st.write(f"Total Strategy Return: **{total_strat:.2%}** over period")
        if "Benchmark" in df_eq.columns:
            total_bench = eq_bench.loc[eq_strat.index[-1]] - 1
            st.write(f"Total Benchmark ({benchmark_choice}) Return: **{total_bench:.2%}** over period")

    # --- Portfolio Optimization Weights ---
    st.subheader("Portfolio Optimization (Efficient Frontier)")
    if len(selected) >= 2:
        solver_opt = st.selectbox("Optimizer Solver EF", ["ECOS", "SCS"], index=0, key="opt_solver_ef")
        try:
            w_sharpe2, w_minvol2 = compute_frontier(data, solver_opt)
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                st.markdown("**Max Sharpe (Tangency) Weights**")
                df_ws = pd.Series(w_sharpe2).sort_values(ascending=False).to_frame("Weight")
                st.table(df_ws.style.format({"Weight": "{:.2%}"}))
            with col_opt2:
                st.markdown("**Min Volatility Weights**")
                df_wm = pd.Series(w_minvol2).sort_values(ascending=False).to_frame("Weight")
                st.table(df_wm.style.format({"Weight": "{:.2%}"}))
        except Exception as e:
            st.error(f"Failed to compute efficient frontier: {e}")
    else:
        st.info("Select at least 2 tickers to compute portfolio optimization.")

if __name__ == "__main__":
    main()
