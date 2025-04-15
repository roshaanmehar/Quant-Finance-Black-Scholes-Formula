import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from options_analyzer import OptionsAnalyzer


st.set_page_config(
    page_title="Options Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'analyzer' not in st.session_state:
    st.session_state.analyzer = OptionsAnalyzer()
    print("Initialized OptionsAnalyzer in session state.")


def format_currency(value, currency='USD'):
    """Helper to format currency within Streamlit"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        # Simplified version for Streamlit, analyzer's version can be used too
        return f"{currency} {value:,.2f}" if currency != 'USD' else f"${value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"
    
    
def display_stock_info(stock_data):
    """Displays formatted stock information."""
    info = stock_data.get('info', {})
    currency = stock_data.get('currency', 'USD')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", format_currency(stock_data['current_price'], currency))
        st.metric("Sector", info.get('sector', 'N/A'))
    with col2:
        if stock_data.get('volatility') is not None:
             st.metric("Hist. Volatility (1Y)", f"{stock_data['volatility']*100:.2f}%")
        else:
             st.metric("Hist. Volatility (1Y)", "N/A")
        st.metric("Industry", info.get('industry', 'N/A'))
    with col3:
        market_cap = info.get('marketCap')
        mc_str = "N/A"
        if market_cap:
            if market_cap >= 1e12: mc_str = f"{format_currency(market_cap / 1e12, currency)}T"
            elif market_cap >= 1e9: mc_str = f"{format_currency(market_cap / 1e9, currency)}B"
            elif market_cap >= 1e6: mc_str = f"{format_currency(market_cap / 1e6, currency)}M"
            else: mc_str = format_currency(market_cap, currency)
        st.metric("Market Cap", mc_str)
        st.metric("Risk-Free Rate", f"{st.session_state.analyzer.risk_free_rate*100:.2f}% (10Y Treasury)")


st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose Analysis:",
    ("Get Quote", "Simple Option Price", "Options Chain Analysis", "Strategy Analysis")
)
st.sidebar.markdown("---")
st.sidebar.write(f"**Risk-Free Rate:** {st.session_state.analyzer.risk_free_rate*100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & yfinance")


if 'stock_data' in st.session_state and st.session_state.stock_data:
    current_ticker = st.session_state.stock_data['ticker']
    st.sidebar.success(f"Current Ticker: **{current_ticker}**")
else:
    st.sidebar.warning("No ticker data loaded.")

if app_mode == "Get Quote":
    st.title("📊 Stock Quote & Data")
    st.markdown("Enter a stock ticker symbol to fetch its current data, volatility, and company information.")

    # Ticker Input Form
    with st.form("ticker_form"):
        ticker_input = st.text_input("Ticker Symbol", placeholder="e.g., AAPL, MSFT, GOOGL").upper()
        submitted = st.form_submit_button("Fetch Data")

        if submitted and ticker_input:
            analyzer = st.session_state.analyzer
            with st.spinner(f"Fetching data for {ticker_input}..."):
                stock_data = analyzer.get_stock_data(ticker_input)
                # Update risk-free rate after potential fetch
                analyzer.get_risk_free_rate()

            if stock_data:
                st.session_state.stock_data = stock_data # Store in session state
                st.success(f"Data loaded for {stock_data['info'].get('shortName', ticker_input)} ({ticker_input})")
            else:
                st.error(f"Could not fetch data for ticker: {ticker_input}. Please check the symbol.")
                if 'stock_data' in st.session_state:
                     del st.session_state.stock_data # Clear old data on failure


