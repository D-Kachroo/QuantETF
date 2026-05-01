# 📊 Full-Stack QuantETF Investment Portfolio Advisor

**Developer:** David Kachroo (CFM @ University of Waterloo)

A full-stack ETF (Exchange-Traded Funds) portfolio advisory and optimization tool that is powered by real-time Yahoo Finance data, AI, machine learning (ML), and various Python libraries/frameworks. Built on VS Code (MacBook) with a Streamlit frontend.

## 🔧 App Features

- A multi-sector ETF selection bar with verified ticker symbols
- Adjusted 'Close' price data from Yahoo Finance
- Risk metrics: volatility, maximum drawdown, CVaR (95%), Sharpe ratio
- Portfolio optimization using Efficient Frontier with L2 regularization
- 30-day return prediction using an ML algorithm (Random Forest)
- Correlation heatmap that compares 2+ ETFs (Pearson's Coefficient)
- Integrating the FRED API for CPI, interest rates, and unemployment rates (USA)
- Downloadable price data (Excel/CSV file)
- Streamlit UI that handles errors and validates user inputs

## ⚙️ Tech Stack

`Python` · `pandas` · `NumPy` · `yfinance` · `QuantLib` · `FRED API` · `PyPortfolioOpt` · `scikit-learn` · `Excel VBA` · `Random Forest (ML)` · `Matplotlib` · `Streamlit` · `Anaconda` · `Docker` · `Railway`

## 🚀 Streamlit Link

```bash
https://quant-etf-advisor.streamlit.app/
