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

    if 'stock_data' in st.session_state and st.session_state.stock_data:
        st.markdown("---")
        st.subheader(f"Details for {st.session_state.stock_data['info'].get('shortName', st.session_state.stock_data['ticker'])}")
        display_stock_info(st.session_state.stock_data)
elif app_mode == "Simple Option Price":
    st.title("💲 Simple Option Price Calculator")
    st.markdown("Calculate theoretical option prices and Greeks using the Black-Scholes-Merton model.")

    if 'stock_data' not in st.session_state or not st.session_state.stock_data:
        st.warning("Please fetch stock data first using the 'Get Quote' section.")
    else:
        stock_data = st.session_state.stock_data
        analyzer = st.session_state.analyzer
        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        expirations = stock_data['expirations']
        currency = stock_data['currency']
        risk_free_rate = analyzer.risk_free_rate

        if not expirations:
             st.error(f"No options expiration dates found for {stock_data['ticker']}.")
        else:
            # Input Form
            with st.form("simple_price_form"):
                st.subheader(f"Parameters for {stock_data['ticker']} ({format_currency(current_price, currency)})")
                col1, col2, col3 = st.columns(3)
                with col1:
                     expiration_date = st.selectbox("Expiration Date", options=expirations)
                with col2:
                     strike = st.number_input("Strike Price", min_value=0.01, value=round(current_price), step=1.0)
                with col3:
                      option_type = st.radio("Option Type", ("Call", "Put", "Both"), horizontal=True)

                # Handle volatility input if needed
                if volatility is None:
                    st.warning("Historical volatility not available. Please provide an estimate.")
                    user_vol = st.number_input("Estimated Annual Volatility (%)", min_value=0.1, value=30.0, step=1.0) / 100.0
                else:
                    user_vol = volatility # Use calculated historical vol

                submitted = st.form_submit_button("Calculate Price")

                if submitted:
                    # Calculate Time to Expiration (T)
                    today = dt.datetime.now().date()
                    exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
                    days_to_expiration = max(0, (exp_date - today).days)
                    T = days_to_expiration / 365.0

                    st.markdown("---")
                    st.subheader("Calculation Results")
                    st.write(f"Using Volatility: {user_vol*100:.2f}%")

                    results_data = []

                    # Calculate Call
                    if option_type in ["Call", "Both"]:
                        bsm_call = analyzer.black_scholes_merton(current_price, strike, T, risk_free_rate, user_vol, "call")
                        greeks_call = analyzer.calculate_option_greeks(current_price, strike, T, risk_free_rate, user_vol, "call")
                        st.markdown(f"**Call Option @ {format_currency(strike, currency)}**")
                        st.metric("BSM Price", format_currency(bsm_call, currency))
                        if greeks_call:
                             with st.expander("Call Greeks", expanded=False):
                                  gc1, gc2 = st.columns(2)
                                  gc1.metric("Delta", f"{greeks_call['delta']:.4f}")
                                  gc1.metric("Gamma", f"{greeks_call['gamma']:.4f}")
                                  gc1.metric("Theta", f"{format_currency(greeks_call['theta'], currency)} / day")
                                  gc2.metric("Vega", f"{format_currency(greeks_call['vega'], currency)} / 1% vol")
                                  gc2.metric("Rho", f"{format_currency(greeks_call['rho'], currency)} / 1% rate")

                    # Calculate Put
                    if option_type in ["Put", "Both"]:
                        bsm_put = analyzer.black_scholes_merton(current_price, strike, T, risk_free_rate, user_vol, "put")
                        greeks_put = analyzer.calculate_option_greeks(current_price, strike, T, risk_free_rate, user_vol, "put")
                        st.markdown(f"**Put Option @ {format_currency(strike, currency)}**")
                        st.metric("BSM Price", format_currency(bsm_put, currency))
                        if greeks_put:
                             with st.expander("Put Greeks", expanded=False):
                                  gp1, gp2 = st.columns(2)
                                  gp1.metric("Delta", f"{greeks_put['delta']:.4f}")
                                  gp1.metric("Gamma", f"{greeks_put['gamma']:.4f}")
                                  gp1.metric("Theta", f"{format_currency(greeks_put['theta'], currency)} / day")
                                  gp2.metric("Vega", f"{format_currency(greeks_put['vega'], currency)} / 1% vol")
                                  gp2.metric("Rho", f"{format_currency(greeks_put['rho'], currency)} / 1% rate")
