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
elif app_mode == "Options Chain Analysis":
    st.title("⛓️ Options Chain Analysis")
    st.markdown("View the options chain for a selected expiration date, comparing market prices with BSM calculations and implied volatility.")

    if 'stock_data' not in st.session_state or not st.session_state.stock_data:
        st.warning("Please fetch stock data first using the 'Get Quote' section.")
    else:
        stock_data = st.session_state.stock_data
        analyzer = st.session_state.analyzer
        expirations = stock_data['expirations']
        currency = stock_data['currency']

        if not expirations:
             st.error(f"No options expiration dates found for {stock_data['ticker']}.")
        else:
            col1, col2 = st.columns([1,3]) # Make selection smaller
            with col1:
                 expiration_date = st.selectbox("Select Expiration Date", options=expirations, key="chain_exp_select")
                 calc_button = st.button("Calculate Chain", key="chain_calc_button")

            if calc_button:
                 with st.spinner(f"Calculating options chain for {expiration_date}..."):
                      # Use a slightly modified call or internal logic for Streamlit
                      # For simplicity, we can reuse the existing method if prints are acceptable during calculation phase
                      # or modify it further to avoid prints.
                      chain_df = analyzer.calculate_options_chain() # Assuming console prints are OK for now

                 if chain_df is not None and not chain_df.empty:
                      st.session_state.chain_df = chain_df # Store for potential visualization
                      st.session_state.chain_exp_date = expiration_date # Store date for vis title
                      st.success(f"Options chain loaded for {expiration_date}.")

                      # Display the chain (using formatting similar to console version if possible)
                      st.subheader(f"Options Chain: {stock_data['ticker']} - {expiration_date}")

                      # Simplified display for Streamlit - can be enhanced
                      st.dataframe(chain_df.style.format({ # Basic formatting
                           'strike': lambda x: format_currency(x, currency),
                           'bsm_call': lambda x: format_currency(x, currency),
                           'market_call': lambda x: format_currency(x, currency),
                           'call_iv': '{:.2f}%',
                           'bsm_put': lambda x: format_currency(x, currency),
                           'market_put': lambda x: format_currency(x, currency),
                           'put_iv': '{:.2f}%',
                           # Add formatting for greeks if shown
                      }, na_rep='N/A'), height=500)


                 else:
                      st.error(f"Could not calculate options chain for {expiration_date}.")
                      if 'chain_df' in st.session_state: del st.session_state.chain_df # Clear old data

            # Option to visualize the currently loaded chain
            if 'chain_df' in st.session_state and st.session_state.chain_df is not None:
                 if st.button("Visualize Options Chain", key="chain_viz_button"):
                      with st.spinner("Generating visualization..."):
                           chain_df_to_plot = st.session_state.chain_df
                           current_price = stock_data['current_price']
                           exp_date_for_plot = st.session_state.chain_exp_date
                           # Call the refactored visualization function that returns a figure
                           fig = analyzer.visualize_options_chain(chain_df_to_plot, current_price, currency, exp_date_for_plot)
                           st.pyplot(fig)
elif app_mode == "Strategy Analysis":
    st.title("📈 Options Strategy Analyzer")
    st.markdown("Analyze the payoff profile, breakeven points, max profit/loss for common options strategies.")

    if 'stock_data' not in st.session_state or not st.session_state.stock_data:
        st.warning("Please fetch stock data first using the 'Get Quote' section.")
    else:
        stock_data = st.session_state.stock_data
        analyzer = st.session_state.analyzer
        S0 = stock_data['current_price']
        expirations = stock_data['expirations']
        currency = stock_data['currency']
        risk_free_rate = analyzer.risk_free_rate
        volatility = stock_data.get('volatility') # May be None

        if not expirations:
             st.error(f"No options expiration dates found for {stock_data['ticker']}. Cannot analyze strategies.")
        else:
            # --- Strategy Selection and Inputs ---
            strategy_map = {
                "Covered Call": 1,
                "Protective Put": 2,
                "Bull Call Spread": 3,
                "Bear Put Spread": 4,
                "Long Straddle": 5,
                "Long Strangle": 6
            }
            strategy_name = st.selectbox("Select Strategy", options=list(strategy_map.keys()))
            strategy_choice = strategy_map[strategy_name]

            with st.form("strategy_form"):
                st.subheader(f"{strategy_name} Parameters")
                expiration_date = st.selectbox("Expiration Date", options=expirations, key="strat_exp")

                # --- Strategy Specific Inputs ---
                K_call, K_put, K_low, K_high = None, None, None, None

                if strategy_choice == 1: # Covered Call
                    K_call = st.number_input("Call Strike (Short)", min_value=0.01, value=round(S0*1.05), step=1.0)
                elif strategy_choice == 2: # Protective Put
                    K_put = st.number_input("Put Strike (Long)", min_value=0.01, value=round(S0*0.95), step=1.0)
                elif strategy_choice == 3: # Bull Call Spread
                    c1, c2 = st.columns(2)
                    K_low = c1.number_input("Lower Call Strike (Long)", min_value=0.01, value=round(S0*0.98), step=1.0)
                    K_high = c2.number_input("Higher Call Strike (Short)", min_value=0.01, value=round(S0*1.02), step=1.0)
                elif strategy_choice == 4: # Bear Put Spread
                    c1, c2 = st.columns(2)
                    K_high = c1.number_input("Higher Put Strike (Long)", min_value=0.01, value=round(S0*1.02), step=1.0)
                    K_low = c2.number_input("Lower Put Strike (Short)", min_value=0.01, value=round(S0*0.98), step=1.0)
                elif strategy_choice == 5: # Long Straddle (ATM)
                    st.write("Straddle uses the strike closest to the current price.")
                    # K_atm determined later
                elif strategy_choice == 6: # Long Strangle
                     c1, c2 = st.columns(2)
                     K_call = c1.number_input("OTM Call Strike (Long)", min_value=0.01, value=round(S0*1.05), step=1.0)
                     K_put = c2.number_input("OTM Put Strike (Long)", min_value=0.01, value=round(S0*0.95), step=1.0)

                submitted = st.form_submit_button("Analyze Strategy")

                # --- Processing ---
                if submitted:
                    strategy_legs = []
                    breakevens = []
                    max_profit = np.nan
                    max_loss = np.nan
                    error_msg = None
                    summary_info = {}

                    try:
                        # Fetch necessary option prices (Handle potential errors/missing data)
                        def get_price(k, opt_type):
                             data = analyzer._get_option_data_for_strike(expiration_date, k, opt_type)
                             if data is None or pd.isna(data['lastPrice']) or data['lastPrice'] <= 0:
                                  st.warning(f"Market price for {opt_type.capitalize()} K={k} unavailable/zero. Using BSM estimate.")
                                  # Fallback BSM calculation (simplified)
                                  vol = volatility if volatility is not None else 0.3 # Use hist vol or 30% default
                                  today = dt.datetime.now().date()
                                  exp_d = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
                                  T = max(0, (exp_d - today).days) / 365.0
                                  price = analyzer.black_scholes_merton(S0, k, T, risk_free_rate, vol, opt_type)
                                  if pd.isna(price): raise ValueError(f"Could not estimate price for {opt_type.capitalize()} K={k}")
                                  return price
                             return data['lastPrice']

                        # --- Build Strategy Legs based on UI inputs ---
                        if strategy_choice == 1: # Covered Call
                             prem_call = get_price(K_call, 'call')
                             strategy_legs = [{'type': 'stock', 'dir': 'long', 'price': S0},
                                              {'type': 'call', 'dir': 'short', 'K': K_call, 'price': prem_call}]
                             cost_basis = S0 - prem_call
                             breakevens = [cost_basis]
                             max_profit = (K_call - S0) + prem_call
                             max_loss = -cost_basis
                             summary_info = {'Net Cost Basis': cost_basis, 'Short Call Premium': prem_call}

                        elif strategy_choice == 2: # Protective Put
                             prem_put = get_price(K_put, 'put')
                             strategy_legs = [{'type': 'stock', 'dir': 'long', 'price': S0},
                                              {'type': 'put', 'dir': 'long', 'K': K_put, 'price': prem_put}]
                             total_cost = S0 + prem_put
                             breakevens = [total_cost]
                             max_profit = float('inf')
                             max_loss = -(S0 - K_put + prem_put)
                             summary_info = {'Total Cost': total_cost, 'Long Put Premium': prem_put}

                        elif strategy_choice == 3: # Bull Call Spread
                             if not (0 < K_low < K_high): raise ValueError("Strikes must be positive and Low < High.")
                             prem_low = get_price(K_low, 'call')
                             prem_high = get_price(K_high, 'call')
                             net_debit = prem_low - prem_high
                             strategy_legs = [{'type': 'call', 'dir': 'long', 'K': K_low, 'price': prem_low},
                                              {'type': 'call', 'dir': 'short', 'K': K_high, 'price': prem_high}]
                             max_profit = (K_high - K_low) - net_debit
                             max_loss = -net_debit
                             breakevens = [K_low + net_debit]
                             summary_info = {'Net Debit': net_debit, 'Long Call Prem': prem_low, 'Short Call Prem': prem_high}

                        elif strategy_choice == 4: # Bear Put Spread
                            if not (0 < K_low < K_high): raise ValueError("Strikes must be positive and Low < High.")
                            prem_high = get_price(K_high, 'put')
                            prem_low = get_price(K_low, 'put')
                            net_debit = prem_high - prem_low
                            strategy_legs = [{'type': 'put', 'dir': 'long', 'K': K_high, 'price': prem_high},
                                              {'type': 'put', 'dir': 'short', 'K': K_low, 'price': prem_low}]
                            max_profit = (K_high - K_low) - net_debit
                            max_loss = -net_debit
                            breakevens = [K_high - net_debit]
                            summary_info = {'Net Debit': net_debit, 'Long Put Prem': prem_high, 'Short Put Prem': prem_low}

                        elif strategy_choice == 5: # Long Straddle
                             # Find ATM strike
                             options = analyzer.current_stock_data['ticker_object'].option_chain(expiration_date)
                             all_strikes = sorted(list(set(options.calls['strike'].tolist() + options.puts['strike'].tolist())))
                             if not all_strikes: raise ValueError("No strikes found.")
                             K_atm = min(all_strikes, key=lambda x: abs(x - S0))
                             st.info(f"Using ATM Strike: {format_currency(K_atm, currency)}")
                             prem_call = get_price(K_atm, 'call')
                             prem_put = get_price(K_atm, 'put')
                             total_cost = prem_call + prem_put
                             strategy_legs = [{'type': 'call', 'dir': 'long', 'K': K_atm, 'price': prem_call},
                                              {'type': 'put', 'dir': 'long', 'K': K_atm, 'price': prem_put}]
                             max_profit = float('inf')
                             max_loss = -total_cost
                             breakevens = [K_atm - total_cost, K_atm + total_cost]
                             summary_info = {'Total Cost': total_cost, 'ATM Strike': K_atm, 'Call Prem': prem_call, 'Put Prem': prem_put}

                        elif strategy_choice == 6: # Long Strangle
                             if not (0 < K_put < K_call): print("Warning: Ensure Call Strike > Put Strike.") # UI allows any K
                             if K_put <= 0 or K_call <= 0: raise ValueError("Strikes must be positive.")
                             prem_call = get_price(K_call, 'call')
                             prem_put = get_price(K_put, 'put')
                             total_cost = prem_call + prem_put
                             strategy_legs = [{'type': 'call', 'dir': 'long', 'K': K_call, 'price': prem_call},
                                              {'type': 'put', 'dir': 'long', 'K': K_put, 'price': prem_put}]
                             max_profit = float('inf')
                             max_loss = -total_cost
                             breakevens = [K_put - total_cost, K_call + total_cost]
                             summary_info = {'Total Cost': total_cost, 'Call Strike': K_call, 'Put Strike': K_put, 'Call Prem': prem_call, 'Put Prem': prem_put}

                    except ValueError as ve:
                        error_msg = f"Input Error: {ve}"
                    except Exception as e:
                         error_msg = f"Analysis Error: {e}"
                         print(f"Strategy analysis error: {e}") # Log for debugging

                    # --- Display Results ---
                    st.markdown("---")
                    if error_msg:
                        st.error(error_msg)
                    elif not strategy_legs:
                         st.error("Could not build strategy legs based on input.")
                    else:
                        st.subheader("Strategy Analysis Results")

                        # Calculate Payoff range
                        price_range_pct = analyzer.config['strategy_price_range']
                        S_T_min = S0 * (1 - price_range_pct)
                        S_T_max = S0 * (1 + price_range_pct)
                        if breakevens:
                            valid_bes = [be for be in breakevens if pd.notna(be)]
                            if valid_bes:
                                S_T_min = min(S_T_min, min(valid_bes) * 0.9)
                                S_T_max = max(S_T_max, max(valid_bes) * 1.1)
                        S_T_range = np.linspace(max(0, S_T_min), S_T_max, 150)
                        PnL = np.array([analyzer._calculate_payoff(s_t, strategy_legs, S0) for s_t in S_T_range])

                        # Display summary table/metrics
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                             st.markdown("**Strategy Legs:**")
                             legs_df = pd.DataFrame(strategy_legs)
                             st.dataframe(legs_df[['dir', 'type', 'K', 'price']].fillna('-').style.format({'K': lambda x: format_currency(x, currency) if x != '-' else '-',
                                                                                                           'price': lambda x: format_currency(x, currency) if x != '-' else '-'}))
                             if summary_info:
                                  st.markdown("**Key Values:**")
                                  for key, val in summary_info.items():
                                       st.metric(key, format_currency(val, currency) if isinstance(val, (int, float)) else val)


                        with col_s2:
                            st.markdown("**Risk Profile:**")
                            be_str = ", ".join([format_currency(be, currency) for be in breakevens if pd.notna(be)]) or "N/A"
                            mp_str = format_currency(max_profit, currency) if pd.notna(max_profit) and max_profit != float('inf') else ('Unlimited' if max_profit == float('inf') else 'N/A')
                            ml_str = format_currency(max_loss, currency) if pd.notna(max_loss) and max_loss != float('-inf') else ('Unlimited' if max_loss == float('-inf') else 'N/A')
                            st.metric("Breakeven(s)", be_str)
                            st.metric("Max Profit", mp_str, delta_color="off") # No up/down arrow needed
                            st.metric("Max Loss", ml_str, delta_color="off")

                        # Display Payoff Plot
                        st.markdown("---")
                        st.subheader("Payoff Diagram")
                        try:
                            # Call refactored plot function
                            fig = analyzer._plot_payoff(S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency)
                            st.pyplot(fig)
                        except Exception as plot_err:
                             st.error(f"Could not generate plot: {plot_err}")