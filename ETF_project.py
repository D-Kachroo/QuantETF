# Full-Stack Quantitative Investment Portfolio Advisor for ETFs
# By: David Kachroo (CFM @ UWaterloo)
# Developed on VS Code as a Streamlit web application
# ======================================================================================================================================
# Overview: This Python script provides a comprehensive tool for ETF investment portfolio management & advisory.
# It's designed to be user-friendly with text, visualizations, formulas, and metrics to help make informed investment decisions.
# It employs Streamlit for the frontend interface, yfinance to retrieve real-time ticker data, and PyPortfolioOpt to optimize ETF portfolios.

# --------------------------------------------------------------------------------------------------------------------------------------
# Imports and Setup Tools + Libraries + Frameworks
# --------------------------------------------------------------------------------------------------------------------------------------
import os
import io
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.objective_functions import L2_reg

# --------------------------------------------------------------------------------------------------------------------------------------
# Federal Reserve Economic Data (FRED) Application Programming Interface (API)
# --------------------------------------------------------------------------------------------------------------------------------------
# Initializes a real-time connection to the FRED API, via a personal access key.
# Example API key:
FRED_API_KEY = "9be2185841963a50ccbf2a2edfb87591"
fred = Fred(api_key=FRED_API_KEY)

# --------------------------------------------------------------------------------------------------------------------------------------
# Loads Valid ETF Metadata from a Local CSV File (etf_list_validated.csv)
# --------------------------------------------------------------------------------------------------------------------------------------
# The valid metadata is used to populate the filter and selection options in the app's sidebar.
etf_metadata = pd.read_csv("etf_list_validated.csv") # 4 columns: Ticker, Name, Sector, Region

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Accesses Global Economic Indicators from FRED
# --------------------------------------------------------------------------------------------------------------------------------------
def get_macro_data():
    try:
        # Latest 10-year treasury constant maturity rate (GS10).
        ir = float(fred.get_series("GS10").dropna().values[-1])
        # Year-over-year percentage change in the Consumer Price Index (CPIAUCSL).
        cpi = float(fred.get_series("CPIAUCSL").pct_change(12).dropna().values[-1] * 100)
        # U.S. unemployment rate (UNRATE).
        ur = float(fred.get_series("UNRATE").dropna().values[-1])
    
    except Exception as e:
        # Displays error in the Streamlit app if FRED data fails to load.
        st.error(f"Macro data fetch failed: {e}")
        ir, cpi, ur = 0.0, 0.0, 0.0

    # Returns a dictionary of the rounded indicators.
    return {
        "Interest Rate (10Y Treasury)": round(ir, 2),
        "Inflation Rate (CPI YoY %)": round(cpi, 2),
        "Unemployment Rate": round(ur, 2),
    }

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Retrieves ETF Data from Yahoo Finance (yfinance)
# --------------------------------------------------------------------------------------------------------------------------------------
# It downloads the price data for the specified ETFs over a given time period and returns a DataFrame with the adjusted close prices.
def get_etf_data(etfs, period="5y"):
    if isinstance(etfs, str):
        etfs = [etfs]

    # Normalizes period input (ensures 'max' is lowercase to comply with yfinance's text requirements)
    period = period.lower()

    # Downloads historical price data with daily frequency and adjustment for dividends/splits.
    data = yf.download(etfs, period=period, interval="1d", auto_adjust=True)

    try:
        if isinstance(data.columns, pd.MultiIndex):
            # Handles multi-ticker format: extracts 'Close' price data from MultiIndex columns.
            adj_close = data["Close"]
        elif "Close" in data.columns:
            # Handles single-ticker format: wraps 'Close' column in a DataFrame and renames column to the ticker.
            adj_close = pd.DataFrame(data["Close"])
            adj_close.columns = etfs
        else:
            raise ValueError("❌ 'Close' column not found in data.")

    except Exception as e:
        raise ValueError(f"❌ Failed to extract adjusted close prices: {e}")

    # Ensures a flat column index (not MultiIndex)
    adj_close.columns.name = None
    adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]  # remove duplicates
    adj_close.columns = [str(col).upper() for col in adj_close.columns]  # just in case

    # Removes any ETF with completely missing data and any remaining missing rows.
    adj_close.dropna(axis=1, how="all", inplace=True)
    adj_close.dropna(inplace=True)

    # Raises an error if no valid ETF data is returned.
    if adj_close.empty:
        raise ValueError("❌ No valid ETF data returned. Please check your tickers and try again.")

    return adj_close

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Calculates and Displays Risk Metrics for the ETF Returns
# --------------------------------------------------------------------------------------------------------------------------------------
# It computes annualized volatility, maximum drawdown, CVaR, and Sharpe ratio. It also categorizes the investment profile based on the SR.
def show_risk_metrics(returns):
    st.subheader("📊 Risk Metrics")

    # Checks if the returns DataFrame is empty or contains NaN values.
    if returns.empty or returns.isna().values.any():
        st.warning("Return data is empty or invalid.")
        return "Unknown"

    try:
        # Annualized Volatility = [standard deviation of daily return] × [sqrt(252 trading days)]
        volatility = float(returns.std().mean() * np.sqrt(252))
        # Conditional Value at Risk (CVaR 95%) = [average of the worst 5% daily returns]
        cvar_95 = float(returns.aggregate(lambda x: np.percentile(x, 5)).mean())
        # Sharpe Ratio = [mean daily return] ÷ [std dev of returns] × [sqrt(252)]
        sharpe = float((returns.mean().mean() / returns.std().mean()) * np.sqrt(252))

        # Maximum Drawdown = [largest peak-to-trough decline in the cumulative returns]
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min().min()

        # Displays all the calculated risk metrics.
        st.metric("📈 Annualized Volatility (Risk)", f"{volatility:.2%}")
        st.metric("💥 Max Drawdown", f"{max_drawdown:.2%}")
        st.metric("📉 Expected Loss in Worst Month (CVaR 95%)", f"{cvar_95:.2%}")
        st.metric("📊 Average Sharpe Ratio", f"{sharpe:.2f}")

        # Categorizes the portfolio's risk profile based on Sharpe Ratio thresholds.
        return "Growth" if sharpe > 1 else "Balanced" if sharpe > 0.5 else "Conservative"

    except Exception as e:
        st.error(f"Risk metric calculation error: {e}")
        return "Unknown"

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Optimizes the ETF(s) Portfolio via the Efficient Frontier Method
# --------------------------------------------------------------------------------------------------------------------------------------
# It uses the PyPortfolioOpt library to calculate the optimal weights for the ETFs based on historical returns and covariance.
# Efficient Frontier is used to maximize the Sharpe ratio, and provide a breakdown of the recommended ETF weightings.
# It also calculates the number of shares to buy based on a $10,000 investment and the latest stock prices.
def optimize_portfolio(price_data):
    st.subheader("⚙️ Portfolio Optimization")
    
    # Calculates expected annual returns (μ) using historical mean returns.
    mu = mean_historical_return(price_data)
    # Computes a covariance matrix (Σ) using the Ledoit-Wolf shrinkage method for better stability.
    S = CovarianceShrinkage(price_data).ledoit_wolf()
    # Initializes the Efficient Frontier optimizer with the expected returns and Σ. The weight bounds are set to (0, 1) to ensure no short selling.
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    # Adds an L2 regularization penalty to discourage large weightings. This is useful in financial portfolios to avoid concentration risk.
    ef.add_objective(L2_reg, gamma=0.1)

    try:
        # Optimizes the portfolio for the maximum Sharpe Ratio (risk-adjusted return).
        weights = ef.max_sharpe()
        # Cleans the raw weight values for display (e.g. removes small decimals).
        cleaned_weights = ef.clean_weights()
        st.write("**Recommended ETF Weights (% of total investment):**")
        st.json(cleaned_weights)
        
        # Retrieves real-time ETF prices to convert weightings into shares.
        latest_prices = get_latest_prices(price_data)

        # Assigns a fixed $10,000 budget using integer programming to determine the ETF(s) share quantities.
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
        allocation, leftover = da.lp_portfolio()

        st.write("**Number of Shares to Buy (based on $10,000):**")
        st.write(allocation)
        st.write(f"Remaining Uninvested Cash: **${leftover:,.2f}**")

    except Exception as e:
        st.error(f"⚠️ Optimization failed: {e}")

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Predicts the Future 30-Day % Return Using a Machine Learning (ML) Model ("Random Forest")
# --------------------------------------------------------------------------------------------------------------------------------------
# It prepares the returns DataFrame, trains a Random Forest ML model, and predicts the next month's ETF returns based on historical prices and economic indicators.
# It handles exceptions and returns None if the prediction fails.
def predict_returns(returns, macro_data):
    try:
        df = returns.copy()
        # Sets the prediction target as the mean return 21 trading days (approx. 1 month) ahead.
        df['Target'] = df.mean(axis=1).shift(-21)
        df.dropna(inplace=True)

        for key, val in macro_data.items():
            df[key] = val

        # Splits data into training and test sets without shuffling (to preserve time order).
        X = df.drop(columns=['Target'])
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Trains a Random Forest Regressor, which handles nonlinear patterns and reduces overfitting.
        model = RandomForestRegressor().fit(X_train, y_train)

        # Evaluates the model's performance on the test set. Uses the final available observation for predictions.
        prediction = model.predict(X_test[-1:])[0]
        return prediction

    except Exception as e:
        st.warning(f"ML prediction failed: {e}")
        return None

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Plots an ETF Correlation Heatmap
# --------------------------------------------------------------------------------------------------------------------------------------
# It uses seaborn to visualize the relationship between different ETFs in a heatmap format.
# The heatmap helps users understand how the ETFs in their portfolio are related to each other.
# It uses Pearson correlation coefficient values, ranging from -1 (strong negative) to +1 (strong positive). A value close to 0 indicates no correlation.
# It allows for identifying pairs of ETFs that perform similarly (++, --) or inversely (+-).
def show_correlation_heatmap(returns):
    # High Value (+) = low diversification/more risk.
    # Low Value (-) = High diversification/less risk.
    st.subheader("📌 Correlation Heatmap (0 = 👍 to 1.00 = 👎)")
    fig, ax = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------------------------------------------------------------------------------------
# Function: Frontend Streamlit Application (User Interface) 
# --------------------------------------------------------------------------------------------------------------------------------------
# It sets up the front-end page, retrieves ETF data, calculates returns, displays risk metrics, and optimizes the portfolio.
# It includes a sidebar for users to input ETF tickers, displays historical prices, and showcases global economic indicators.
def main():
    st.set_page_config(page_title="Streamlit Quant Project (ETFs)", layout="wide")
    st.title("💼 Investment Portfolio Advisor (QuantETF) - By: David K.")

    # Sidebar configuration: filter by sector and select ETFs/time period.
    with st.sidebar:
        st.header("ETF Selection (Sidebar)")
        sector_choice = st.selectbox("Filter by sector:", options=etf_metadata['Sector'].unique())
        filtered = etf_metadata[etf_metadata['Sector'] == sector_choice]
        selected_etfs = st.multiselect("Choose your ETF(s):", options=filtered['Ticker'])
        time_horizon = st.selectbox("Select a time period:", ["1y", "3y", "5y", "10y", "max"])
        st.caption("Recommendation: Use multi-sector ETFs for diversification.")

    if not selected_etfs:
        st.warning("Please select at least one ETF.")
        return

    try:
        # Loads and cleans historical price data based on user input.
        price_data = get_etf_data(selected_etfs, period=time_horizon)
        st.write("✅ Valid ETFs:", list(price_data.columns))

        # Calculates daily percentage returns.
        returns = price_data.pct_change().dropna()

        st.subheader("📈 Historical Price Chart")
        st.line_chart(price_data)

        with st.expander("🔍 View: 5-Day Returns Data"):
            st.dataframe(returns.tail())

        # Displays the risk metrics and investor profile classification.
        profile = show_risk_metrics(returns)
        st.success(f"Investment Category: **{profile}**")

        # Displays a correlation matrix (heatmap).
        show_correlation_heatmap(returns)

        # Displays macroeconomic indicators from FRED.
        st.subheader("🌍 Global Economic Indicators")
        macro_data = get_macro_data()
        for label, value in macro_data.items():
            suffix = "%" if "Rate" in label or "CPI" in label else ""
            st.metric(label, f"{value:.2f}{suffix}")

        # Forecasts next 30-day return using ML model (Random Forest).
        prediction = predict_returns(returns, macro_data)
        if prediction is not None:
            st.metric("📈 Machine Learning (ML) Forecasted Return (next 30d)", f"{prediction*100:.2f}%")

        # Optimizes user's ETF portfolio using the Efficient Frontier method.
        optimize_portfolio(price_data)

        # Allows user to export historical price data from the selected time period.
        buffer = io.BytesIO()
        price_data.to_excel(buffer)
        st.download_button("📁 Download Historical Prices Data (.csv)", buffer.getvalue(), file_name="etf_prices.xlsx")

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
