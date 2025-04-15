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







