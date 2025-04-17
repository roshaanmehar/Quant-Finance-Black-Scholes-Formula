# options_analyzer.py
# Enhanced Options Pricing and Analysis Tool (Backend)

# Dependencies: numpy, pandas, yfinance, matplotlib, scipy, tabulate
# Install using: pip install numpy pandas yfinance matplotlib scipy tabulate

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.stats import norm
# from scipy.optimize import brentq # Alternative for IV calculation
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import style
import os
import json
from time import sleep
import warnings

# Suppress specific warnings (e.g., from yfinance or plotting)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="Parsing dates in DD/MM/YYYY format is deprecated")

# Set plotting style
style.use('ggplot')

class OptionsAnalyzer:
    """
    A tool for fetching stock data, calculating option prices (Black-Scholes-Merton),
    analyzing option chains, calculating Greeks, implied volatility, and evaluating
    common option strategies with payoff diagrams. Backend logic for console or UI.
    """
    def __init__(self):
        """Initialize the Options Analyzer with default parameters"""
        self.current_ticker = None
        self.current_stock_data = None
        self.risk_free_rate = None
        self.config = self._load_config()
        self.favorite_tickers = self._load_favorite_tickers()
        # Fetch initial risk-free rate silently for backend/UI use
        self._fetch_risk_free_rate_silently() # Changed from get_risk_free_rate

    # --- Configuration and Persistence ---

    def _load_config(self):
        """Load configuration from file or use defaults."""
        default_config = {
            'volatility_days': 252,
            'default_risk_free_rate': 0.04,
            'show_greeks_in_chain': True,
            'max_strikes_chain': 15,
            'iv_precision': 0.0001,
            'iv_max_iterations': 100,
            'strategy_price_range': 0.3,
            'debug_mode': False
        }
        config_path = 'options_config.json'
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    for key, value in default_config.items():
                        if key not in config: config[key] = value
                    # print("Configuration loaded.") # Keep console clean for UI use
                    return config
            else:
                # print("No config file found, using defaults.")
                return default_config
        except Exception as e:
            print(f"Warning: Error loading config: {e}. Using defaults.") # Keep warnings
            return default_config

    def _save_config(self):
        """Save configuration to file."""
        config_path = 'options_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            # print("Configuration saved successfully.") # Keep console clean
        except Exception as e:
            print(f"Warning: Error saving configuration: {e}")

    def _load_favorite_tickers(self):
        """Load favorite tickers from file."""
        favorites_path = 'favorite_tickers.json'
        try:
            if os.path.exists(favorites_path):
                with open(favorites_path, 'r') as f:
                    favorites = json.load(f)
                    # print(f"Loaded {len(favorites)} favorite tickers.")
                    return favorites
            else:
                 # print("No favorite tickers file found.")
                 return []
        except Exception as e:
            print(f"Warning: Error loading favorite tickers: {e}. Starting fresh.")
            return []

    def _save_favorite_tickers(self):
        """Save favorite tickers to file."""
        favorites_path = 'favorite_tickers.json'
        try:
            with open(favorites_path, 'w') as f:
                json.dump(self.favorite_tickers, f, indent=4)
            # print("Favorite tickers saved successfully.")
        except Exception as e:
            print(f"Warning: Error saving favorite tickers: {e}")

    # --- Utility Functions ---

    def clear_screen(self):
        """Clear the console screen (used only in console mode)."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _format_currency(self, value, currency='USD'):
        """Formats a numeric value as currency."""
        if pd.isna(value): return "N/A"
        try:
            if currency == 'USD': return f"${value:,.2f}"
            else: return f"{value:,.2f} {currency}"
        except (TypeError, ValueError): return "N/A"

    def validate_ticker(self, ticker):
        """Validate if the ticker exists using yfinance (minimal printing)."""
        if not ticker: return False
        try:
            # print(f"\nValidating ticker '{ticker}'...") # Keep console clean
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or info.get('quoteType') == 'MUTUALFUND':
                 hist = stock.history(period="5d")
                 if hist.empty: return False
            if info.get('quoteType') not in ['EQUITY', 'ETF']:
                 print(f"Warning: Ticker '{ticker}' may not have options ({info.get('quoteType')}).")
            # print(f"Ticker '{ticker}' appears valid.") # Keep console clean
            return True
        except Exception as e:
            # print(f"Error validating ticker '{ticker}': {e}") # Keep console clean
            if self.config['debug_mode']: print(f"Debug (validate_ticker): {e}")
            return False

    def _select_expiration_date(self, expirations):
        """Lists available expiration dates and prompts user selection (CONSOLE ONLY)."""
        # This method is problematic for UI use due to input().
        # It should ideally not be called directly from the Streamlit app.
        # The Streamlit app should pass the selected date directly.
        if not expirations:
            print("No expiration dates available for this ticker.")
            return None

        print("\nAvailable expiration dates:")
        valid_expirations = []
        today = dt.datetime.now().date()
        for i, date_str in enumerate(expirations):
            try:
                exp_date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
                days = (exp_date - today).days
                if days >= 0:
                    print(f"{len(valid_expirations) + 1}. {date_str} ({days} days)")
                    valid_expirations.append({'index': i, 'date': date_str, 'days': days})
            except ValueError: continue

        if not valid_expirations:
            print("No valid future expiration dates found.")
            return None

        while True: # Loop only makes sense in console mode
            try:
                selection = input(f"\nSelect expiration date (1-{len(valid_expirations)}): ")
                if not selection:
                    selected_exp = valid_expirations[0]; break
                idx = int(selection) - 1
                if 0 <= idx < len(valid_expirations):
                    selected_exp = valid_expirations[idx]; break
                else: print("Invalid selection.")
            except ValueError: print("Invalid input.")

        print(f"\nSelected expiration date: {selected_exp['date']} ({selected_exp['days']} days)")
        return selected_exp['date']

    # --- Data Fetching ---

    def get_stock_data(self, ticker):
        """Fetch stock data, company info, and options expirations (minimal printing)."""
        ticker = ticker.upper().strip()
        # Basic validation without print unless debug
        if not self.validate_ticker(ticker):
            return None # Validation failed

        try:
            # print(f"\nFetching data for {ticker}...") # Keep console clean
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            current_price = None
            volatility = None

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                if len(returns) >= 2:
                     volatility = returns.std() * np.sqrt(self.config['volatility_days'])
                else:
                     print(f"Warning: Not enough history for {ticker} volatility calculation.")

            # Try getting info even if history failed (for current price fallback)
            info = {}
            currency = 'USD'
            dividend_yield = 0.0 # Default to 0
            try:
                info = stock.info
                currency = info.get('currency', 'USD')
                dividend_yield = info.get('dividendYield') or 0.0 # Fetch dividend yield, default 0 if None
                if current_price is None: # Fallback price
                     current_price = info.get('currentPrice') or info.get('previousClose')
            except Exception as e_info:
                 if self.config['debug_mode']: print(f"Debug (get_stock_data info): {e_info}")
                 # If history failed AND info failed, we have no price
                 if current_price is None: raise ValueError("Could not fetch price via history or info.")

            if current_price is None: # Should be caught above, but final check
                 raise ValueError(f"Failed to determine current price for {ticker}.")


            # Get expirations silently
            expirations = ()
            try: expirations = stock.options
            except Exception: pass # Ignore errors here, handled later if needed

            self.current_ticker = ticker
            self.current_stock_data = {
                'ticker': ticker,
                'current_price': current_price,
                'volatility': volatility, # Can be None
                'dividend_yield': float(dividend_yield) if dividend_yield is not None else 0.0, # Store yield
                'expirations': expirations,
                'ticker_object': stock,
                'history': hist,
                'info': info,
                'currency': currency
            }
            # print(f"Data fetch complete for {ticker}.") # Keep console clean
            return self.current_stock_data

        except Exception as e:
            print(f"\nError fetching data for '{ticker}': {e}") # Keep essential errors
            if self.config['debug_mode']: import traceback; traceback.print_exc()
            if self.current_ticker == ticker: self.current_ticker = None; self.current_stock_data = None
            return None

    def _fetch_risk_free_rate_silently(self):
        """Fetches risk-free rate with minimal printing, suitable for backend."""
        try:
            treasury = yf.Ticker("^TNX") # 10-Year Treasury Yield
            data = treasury.history(period="5d")
            if not data.empty:
                rate = data['Close'].iloc[-1] / 100
                if 0 <= rate <= 0.2:
                    # print(f"Using current risk-free rate: {rate:.4f}") # Keep console clean
                    self.risk_free_rate = rate
                    return rate
        except Exception as e:
             if self.config['debug_mode']: print(f"Debug (fetch_risk_free_rate): {e}")
             pass # Fallback silently

        default_rate = self.config['default_risk_free_rate']
        # print(f"Using default risk-free rate: {default_rate:.4f}") # Keep console clean
        self.risk_free_rate = default_rate
        return default_rate

    def get_risk_free_rate(self):
        """Public method to get/refresh risk-free rate (can print for console)."""
        # This method can be used by the console runner if more verbosity is desired.
        # The UI should rely on the rate fetched silently at init or refreshed internally.
        print("Fetching current risk-free rate (10-Year Treasury Yield)...")
        rate = self._fetch_risk_free_rate_silently()
        if rate == self.config['default_risk_free_rate']:
             print(f"Using default risk-free rate: {rate:.4f} ({rate*100:.2f}%)")
        else:
             print(f"Using current risk-free rate: {rate:.4f} ({rate*100:.2f}%)")
        return rate


    def _get_option_data_for_strike(self, expiration_date, strike, option_type):
         """Helper to get specific option data (call or put) for a strike."""
         if not self.current_stock_data or not self.current_stock_data.get('ticker_object'):
             print("Error: Stock data not loaded.")
             return None
         if not expiration_date: return None

         stock = self.current_stock_data['ticker_object']
         option_type = option_type.lower()

         try:
             opt_chain = stock.option_chain(expiration_date)
             data = opt_chain.calls if option_type == 'call' else opt_chain.puts
             option_data = data[data['strike'] == strike]
             if option_data.empty:
                 # print(f"Warning: No {option_type} data for K={strike} on {expiration_date}.") # Keep console clean
                 return None
             return option_data.iloc[0] # Return Series
         except IndexError: return None # Handle empty structures
         except Exception as e:
             print(f"Error fetching option data K={strike} ({option_type}): {e}")
             if self.config['debug_mode']: import traceback; traceback.print_exc()
             return None

    # --- Black-Scholes-Merton Model and Greeks (with Dividend Yield q) ---

    def black_scholes_merton(self, S, K, T, r, q, sigma, option_type="call"):
        """ BSM Option Price including continuous dividend yield q. """
        if T < 0: T = 0
        if sigma <= 0: sigma = 1e-6
        if S <=0 or K <= 0: return np.nan

        if T == 0:
            return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        try:
            if option_type.lower() == "call":
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == "put":
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            else: return np.nan
            return max(0.0, price)
        except (OverflowError, ValueError): return np.nan # Catch potential math errors
        except Exception as e: print(f"Error in BSM: {e}"); return np.nan

    def calculate_option_greeks(self, S, K, T, r, q, sigma, option_type="call"):
        """ Calculate option Greeks including continuous dividend yield q. """
        greeks = { "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan }
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return greeks

        option_type = option_type.lower()
        try:
            sqrt_T = np.sqrt(T)
            exp_qT = np.exp(-q * T)
            exp_rT = np.exp(-r * T)

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_neg_d2 = norm.cdf(-d2)

            greeks["gamma"] = exp_qT * pdf_d1 / (S * sigma * sqrt_T) if (S * sigma * sqrt_T) != 0 else 0
            greeks["vega"] = (S * exp_qT * sqrt_T * pdf_d1) / 100 # per 1% change

            # Theta (per day)
            theta_term1 = - (S * exp_qT * pdf_d1 * sigma) / (2 * sqrt_T)

            if option_type == "call":
                greeks["delta"] = exp_qT * cdf_d1
                theta_term2 = - r * K * exp_rT * cdf_d2
                theta_term3 = + q * S * exp_qT * cdf_d1
                greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
                greeks["rho"] = (K * T * exp_rT * cdf_d2) / 100 # per 1% change in r
            elif option_type == "put":
                greeks["delta"] = exp_qT * (cdf_d1 - 1)
                theta_term2 = + r * K * exp_rT * cdf_neg_d2
                theta_term3 = - q * S * exp_qT * cdf_neg_d1
                greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
                greeks["rho"] = (-K * T * exp_rT * cdf_neg_d2) / 100 # per 1% change in r
            else: return {k: np.nan for k in greeks}

            return greeks

        except (ZeroDivisionError, OverflowError, ValueError): return {k: np.nan for k in greeks} # Return NaNs on math errors
        except Exception as e: print(f"Error calculating Greeks: {e}"); return {k: np.nan for k in greeks}

    def calculate_implied_volatility(self, S, K, T, r, q, market_price, option_type="call"):
        """ Calculate implied volatility using bisection (includes dividend yield q). """
        option_type = option_type.lower()
        precision = self.config['iv_precision']
        max_iterations = self.config['iv_max_iterations']

        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0: return np.nan

        intrinsic_value = 0.0
        if option_type == "call": intrinsic_value = max(0.0, S * np.exp(-q*T) - K * np.exp(-r * T)) # Compare with discounted intrinsic
        elif option_type == "put": intrinsic_value = max(0.0, K * np.exp(-r * T) - S * np.exp(-q*T))
        else: return np.nan

        if market_price < intrinsic_value - precision: return np.nan # Price below intrinsic

        vol_low, vol_high = 1e-5, 5.0
        price_low = self.black_scholes_merton(S, K, T, r, q, vol_low, option_type)
        price_high = self.black_scholes_merton(S, K, T, r, q, vol_high, option_type)

        if pd.isna(price_low) or pd.isna(price_high): return np.nan # Error at bounds
        if market_price <= price_low: return vol_low
        if market_price >= price_high: return vol_high

        for _ in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            price_mid = self.black_scholes_merton(S, K, T, r, q, vol_mid, option_type)

            if pd.isna(price_mid): # Handle errors during iteration
                 # This might indicate instability; often happens with extreme inputs.
                 # Returning NaN is safer than potentially converging to a wrong value.
                 return np.nan

            if abs(price_mid - market_price) < precision: return vol_mid
            if price_mid > market_price: vol_high = vol_mid
            else: vol_low = vol_mid
            if abs(vol_high - vol_low) < precision: break

        final_vol = (vol_low + vol_high) / 2
        final_price = self.black_scholes_merton(S, K, T, r, q, final_vol, option_type)
        # Check if final result is reasonably close before returning
        if pd.notna(final_price) and abs(final_price - market_price) < precision * 10:
             return final_vol
        else: return np.nan # Failed to converge reliably

    # --- Core Functionality Methods ---

    def get_simple_option_price(self, ticker=None):
        """Calculate and display a simple option price (CONSOLE USE)."""
        # This method uses input() and print() extensively, suitable for console runner.
        if ticker is None:
            if self.current_ticker is None:
                ticker = input("Enter stock ticker symbol: ").upper()
                if not ticker: return
            else:
                ticker = self.current_ticker

        # Get data (prints details)
        stock_data = self.get_stock_data(ticker)
        if not stock_data: return

        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        dividend_yield = stock_data['dividend_yield'] # Get yield
        expirations = stock_data['expirations']
        stock = stock_data['ticker_object']
        currency = stock_data['currency']
        risk_free_rate = self.risk_free_rate # Use fetched rate

        if volatility is None:
             print("\nWarning: Historical volatility is not available. Using 0 unless provided.")
             try: user_vol = float(input("Enter estimated annual volatility (e.g., 0.3): ") or 0.0)
             except ValueError: user_vol = 0.0
             volatility = user_vol if user_vol > 0 else 1e-6

        # Select Expiration (uses console input)
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date: return

        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days)
        T = days_to_expiration / 365.0
        print(f"Time to expiration (T): {T:.4f} years")

        # Select Strike (uses console input)
        strike = None
        while strike is None:
            strike_input = input(f"Enter strike price ('atm' for closest ~{current_price:.2f}): ").lower()
            if strike_input == 'atm':
                try:
                    options = stock.option_chain(expiration_date)
                    all_strikes = sorted(list(set(options.calls['strike'].tolist() + options.puts['strike'].tolist())))
                    strike = min(all_strikes, key=lambda x: abs(x - current_price)) if all_strikes else current_price
                    print(f"Using closest available strike: {self._format_currency(strike, currency)}")
                except Exception as e: strike = current_price; print(f"Could not fetch strikes. Using current price. Err:{e}")
            else:
                try: strike = float(strike_input); assert strike > 0
                except (ValueError, AssertionError): print("Invalid strike.")

        # Select Option Type (uses console input)
        option_type = None
        while option_type not in ['call', 'put', 'both']:
            opt_in = input("Calculate for 'call', 'put', or 'both'? (both): ").lower() or 'both'
            if opt_in in ['call', 'put', 'both']: option_type = opt_in

        # Calculate and Display (prints results)
        results = {}
        print(f"\n--- BSM Option Analysis ---") # Print header
        print(f"Stock: {self.current_ticker} @ {self._format_currency(current_price, currency)}")
        print(f"Div Yield: {dividend_yield*100:.2f}%") # Show yield
        # ... print other params ...

        if option_type in ['call', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "call")
            greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "call")
            # ... print call results ...
            print(f"BSM Call Price: {self._format_currency(bsm_price, currency)}") # Example print
            # ... print greeks ...

        if option_type in ['put', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "put")
            greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "put")
            # ... print put results ...
            print(f"BSM Put Price: {self._format_currency(bsm_price, currency)}") # Example print
            # ... print greeks ...


    def calculate_options_chain(self, ticker=None, specific_expiration=None):
        """
        Calculate and return a detailed options chain DataFrame.
        Minimal printing, suitable for UI or backend use.
        Accepts an optional specific_expiration date.
        """
        # --- Simplified Ticker/Data Handling ---
        if ticker is None:
            if self.current_ticker is None:
                print("Error: No ticker specified and no current ticker set.")
                return None
            ticker = self.current_ticker

        # Fetch/use data for the specified ticker
        # Use get_stock_data which now includes dividend yield
        if self.current_stock_data is None or self.current_ticker != ticker:
             stock_data = self.get_stock_data(ticker)
             if not stock_data: return None # Error handled in get_stock_data
        else:
             stock_data = self.current_stock_data # Use existing loaded data

        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        dividend_yield = stock_data['dividend_yield'] # Fetch yield
        expirations = stock_data.get('expirations', ())
        stock = stock_data['ticker_object']
        currency = stock_data.get('currency', 'USD')
        risk_free_rate = self.risk_free_rate # Use current rate

        # --- Determine Expiration Date ---
        expiration_date = None
        if specific_expiration: # Use passed expiration if provided
             if specific_expiration in expirations:
                 expiration_date = specific_expiration
                 # print(f"\nUsing specified expiration date: {expiration_date}") # Keep console clean
             else:
                  print(f"Error: Provided expiration '{specific_expiration}' not valid for {ticker}.")
                  return None
        else: # If no specific date, default to first available (or could prompt in console mode)
            if expirations:
                 expiration_date = expirations[0]
                 # print(f"Warning: No specific expiration provided. Using first available: {expiration_date}") # Keep clean
            else:
                 print(f"Error: No options expiration dates found for {ticker}.")
                 return None

        # --- Basic Checks ---
        if volatility is None:
             print("\nWarning: Historical volatility unavailable. BSM prices will use 0 vol.")
             volatility = 1e-6 # Use tiny vol as default if missing

        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days)
        T = days_to_expiration / 365.0
        # print(f"Time to expiration (T): {T:.4f} years") # Keep clean

        # --- Fetch Option Chain Data ---
        try:
            # print("\nFetching options chain data...") # Keep clean
            options = stock.option_chain(expiration_date)
            calls = options.calls
            puts = options.puts
            if calls.empty and puts.empty: return pd.DataFrame() # Return empty DF

            # print("Calculating BSM prices, IV, and Greeks...") # Keep clean

            calls = calls.add_prefix('call_').rename(columns={'call_strike': 'strike'})
            puts = puts.add_prefix('put_').rename(columns={'put_strike': 'strike'})
            chain_df = pd.merge(calls, puts, on='strike', how='outer').sort_values(by='strike').reset_index(drop=True)

            # Limit Strikes Around ATM
            max_strikes = self.config['max_strikes_chain']
            if len(chain_df) > max_strikes:
                atm_index = chain_df.iloc[(chain_df['strike'] - current_price).abs().argsort()[:1]].index[0]
                half_width = max_strikes // 2
                start_idx = max(0, atm_index - half_width)
                end_idx = min(len(chain_df), start_idx + max_strikes)
                if (end_idx - start_idx) < max_strikes: start_idx = max(0, end_idx - max_strikes)
                chain_df = chain_df.iloc[start_idx:end_idx].reset_index(drop=True)
                # print(f"Displaying {len(chain_df)} strikes...") # Keep clean

            # --- Calculate BSM, IV, Greeks ---
            results = []
            for _, row in chain_df.iterrows():
                strike = row['strike']
                data = {'strike': strike}
                market_call = row.get('call_lastPrice')
                market_put = row.get('put_lastPrice')

                # Use calculated IV for BSM if available, else use historical/input vol
                call_iv = np.nan
                put_iv = np.nan
                if pd.notna(market_call):
                     call_iv = self.calculate_implied_volatility(current_price, strike, T, risk_free_rate, dividend_yield, market_call, "call")
                if pd.notna(market_put):
                      put_iv = self.calculate_implied_volatility(current_price, strike, T, risk_free_rate, dividend_yield, market_put, "put")

                vol_to_use_call = call_iv if pd.notna(call_iv) else volatility
                vol_to_use_put = put_iv if pd.notna(put_iv) else volatility

                # Call Calcs
                data['market_call'] = market_call
                data['call_iv'] = call_iv * 100 if pd.notna(call_iv) else np.nan
                data['bsm_call'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_call, "call")
                if self.config['show_greeks_in_chain']:
                     greeks_call = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_call, "call")
                     data.update({f'call_{k}': v for k, v in greeks_call.items()}) # Add greeks

                # Put Calcs
                data['market_put'] = market_put
                data['put_iv'] = put_iv * 100 if pd.notna(put_iv) else np.nan
                data['bsm_put'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_put, "put")
                if self.config['show_greeks_in_chain']:
                      greeks_put = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_put, "put")
                      data.update({f'put_{k}': v for k, v in greeks_put.items()}) # Add greeks

                # Other data
                data['call_volume'] = row.get('call_volume')
                data['call_oi'] = row.get('call_openInterest')
                data['put_volume'] = row.get('put_volume')
                data['put_oi'] = row.get('put_openInterest')
                results.append(data)

            results_df = pd.DataFrame(results)

            # ***** INPUT PROMPT REMOVED *****
            # The decision to visualize is now handled ONLY by the frontend (app.py)
            # It will call visualize_options_chain separately if requested.

            return results_df # Return the calculated data

        except AttributeError:
             print(f"Error: Could not find options data structure for {ticker} on {expiration_date}.")
             return None # Return None on specific yfinance structure errors
        except Exception as e:
            print(f"Error calculating options chain: {e}")
            if self.config['debug_mode']: import traceback; traceback.print_exc()
            return None # Return None on general errors


    def visualize_options_chain(self, df, current_price, currency, expiration_date):
        """Visualize the options chain data. Returns matplotlib figure."""
        if df is None or df.empty: return None
        df = df.copy().dropna(subset=['strike'])
        if df.empty: return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{self.current_ticker} Options ({expiration_date}) | Stock: {self._format_currency(current_price, currency)}", fontsize=14)

        ax1 = axes[0]
        if 'market_call' in df.columns: ax1.plot(df['strike'], df['market_call'], 'bo-', label='Market Call', markersize=4, alpha=0.7)
        if 'bsm_call' in df.columns: ax1.plot(df['strike'], df['bsm_call'], 'b--', label='BSM Call', alpha=0.6)
        if 'market_put' in df.columns: ax1.plot(df['strike'], df['market_put'], 'ro-', label='Market Put', markersize=4, alpha=0.7)
        if 'bsm_put' in df.columns: ax1.plot(df['strike'], df['bsm_put'], 'r--', label='BSM Put', alpha=0.6)
        ax1.set_ylabel(f'Option Price ({currency})'); ax1.set_title('Market vs. BSM Price'); ax1.grid(True);
        ax1.axvline(current_price, color='grey', linestyle=':', lw=1.5, label=f'Stock Price')
        ax1.legend(fontsize=9)

        ax2 = axes[1]
        if 'call_iv' in df.columns and df['call_iv'].notna().any(): ax2.plot(df['strike'], df['call_iv'], 'go-', label='Call IV (%)', markersize=4)
        if 'put_iv' in df.columns and df['put_iv'].notna().any(): ax2.plot(df['strike'], df['put_iv'], 'mo-', label='Put IV (%)', markersize=4)
        if self.current_stock_data and self.current_stock_data['volatility'] is not None:
             hist_vol = self.current_stock_data['volatility'] * 100
             ax2.axhline(hist_vol, color='black', linestyle='--', lw=1, label=f'Hist. Vol ({hist_vol:.2f}%)')
        ax2.set_xlabel('Strike Price'); ax2.set_ylabel('Implied Volatility (%)'); ax2.set_title('Implied Volatility Smile / Skew'); ax2.grid(True);
        ax2.axvline(current_price, color='grey', linestyle=':', lw=1.5)
        ax2.legend(fontsize=9)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig # Return the figure object


    # --- Options Strategy Analysis (Payoff Calculation and Plotting) ---

    def _calculate_payoff(self, S_T, strategy_legs, S0):
        """Calculates the P/L of a strategy at expiration price S_T."""
        # Note: S0 (price at analysis time) isn't strictly needed for terminal payoff,
        # but useful if including cost basis calculation within the payoff function.
        # Current implementation calculates cost separately.
        total_payoff = 0
        initial_cost = sum(leg['price'] for leg in strategy_legs if leg['dir'] == 'long')
        initial_credit = sum(leg['price'] for leg in strategy_legs if leg['dir'] == 'short')
        net_cost_or_credit = initial_cost - initial_credit

        for leg in strategy_legs:
            leg_type, direction, K = leg['type'], leg['dir'], leg.get('K')
            payoff_leg = 0
            if leg_type == 'stock': payoff_leg = S_T # Payoff relative to 0, cost handled below
            elif leg_type == 'call': payoff_leg = max(0, S_T - K)
            elif leg_type == 'put': payoff_leg = max(0, K - S_T)

            if direction == 'short': payoff_leg *= -1
            total_payoff += payoff_leg

        # P/L = Final Value of Position - Net Initial Cost (or + Net Initial Credit)
        profit_loss = total_payoff - net_cost_or_credit
        return profit_loss

    def _plot_payoff(self, S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency):
        """Plots the P/L diagram. Returns matplotlib figure."""
        fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes object
        ax.plot(S_T_range, PnL, lw=2, label='Profit/Loss at Expiration')
        ax.axhline(0, color='black', linestyle='--', lw=1, label='Breakeven Level')

        valid_bes = [be for be in breakevens if pd.notna(be)]
        if valid_bes:
            ax.scatter(valid_bes, [0] * len(valid_bes), color='red', s=80, zorder=5, label=f'Breakeven(s): {", ".join([self._format_currency(be, currency) for be in valid_bes])}')

        if pd.notna(max_profit) and max_profit != float('inf'):
            ax.axhline(max_profit, color='green', linestyle=':', lw=1, label=f'Max Profit: {self._format_currency(max_profit, currency)}')
        if pd.notna(max_loss) and max_loss != float('-inf'):
             ax.axhline(max_loss, color='red', linestyle=':', lw=1, label=f'Max Loss: {self._format_currency(max_loss, currency)}')

        ax.set_title(f'{strategy_name} Payoff Diagram'); ax.set_xlabel(f'Underlying Price at Expiration ({currency})'); ax.set_ylabel(f'Profit / Loss ({currency})');
        ax.grid(True); ax.legend(fontsize=9);

        # Adjust y-limits to ensure max profit/loss lines are visible
        y_min = min(PnL.min(), max_loss if pd.notna(max_loss) and max_loss != float('-inf') else PnL.min())
        y_max = max(PnL.max(), max_profit if pd.notna(max_profit) and max_profit != float('inf') else PnL.max())
        padding = (y_max - y_min) * 0.1 # 10% padding
        ax.set_ylim(y_min - padding, y_max + padding)

        return fig # Return the figure


    def analyze_strategy(self):
        """Guides user through selecting and analyzing an options strategy (CONSOLE USE)."""
        # This method uses input() and print() and should primarily be used by the console runner.
        # The Streamlit app (`app.py`) replicates this logic using UI widgets.
        if self.current_stock_data is None:
            print("\nPlease fetch stock data first.")
            ticker = input("Enter stock ticker symbol: ").upper()
            if not ticker or not self.get_stock_data(ticker): return
        # else: # No need to ask to switch ticker here for console version logic flow

        stock_data = self.current_stock_data
        S0 = stock_data['current_price']
        expirations = stock_data['expirations']
        currency = stock_data['currency']
        dividend_yield = stock_data['dividend_yield'] # Get yield
        risk_free_rate = self.risk_free_rate
        volatility = stock_data['volatility'] # May be None

        print("\n--- Options Strategy Analysis ---")
        # ... (print strategy menu as before) ...
        # ... (get strategy_choice via input() as before) ...
        # ... (call _select_expiration_date() as before) ...
        # ... (get strategy parameters K_call, K_put etc via input() as before) ...

        # Inside the try block after getting parameters:
        try:
            # Helper to get price, potentially using BSM fallback (CONSOLE VERSION)
            def get_price_console(k, opt_type, exp_date):
                data = self._get_option_data_for_strike(exp_date, k, opt_type)
                if data is not None and pd.notna(data['lastPrice']) and data['lastPrice'] > 0:
                    return data['lastPrice']
                else:
                    print(f"Warning: Market price for {opt_type.capitalize()} K={k} unavailable. Using BSM estimate.")
                    vol = volatility if volatility is not None else 0.3
                    today = dt.datetime.now().date()
                    exp_d = dt.datetime.strptime(exp_date, '%Y-%m-%d').date()
                    T_est = max(0, (exp_d - today).days) / 365.0
                    price = self.black_scholes_merton(S0, k, T_est, risk_free_rate, dividend_yield, vol, opt_type)
                    if pd.isna(price): raise ValueError(f"Could not estimate price for {opt_type.capitalize()} K={k}")
                    print(f"BSM Estimated Price: {self._format_currency(price, currency)}")
                    return price

            # ... (build strategy_legs using get_price_console as before) ...
            # ... (calculate breakevens, max_profit, max_loss as before) ...

            # ... (print summary as before) ...

            # ... (calculate S_T_range and PnL as before) ...

            # Plot Payoff Diagram (and display using plt.show() for console)
            fig = self._plot_payoff(S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency)
            plt.show() # Show plot in console mode

        except ValueError as ve: print(f"\nInput Error: {ve}")
        except Exception as e: print(f"\nAn error occurred: {e}"); # traceback optional

    # --- Menu and Application Flow (CONSOLE USE) ---

    def manage_favorites(self):
        """Manage favorite tickers (CONSOLE USE)."""
        # ... (Implementation uses print and input - suitable for console) ...
        pass # Placeholder - keep existing console logic if needed

    def manage_settings(self):
         """Manage settings (CONSOLE USE)."""
         # ... (Implementation uses print and input - suitable for console) ...
         pass # Placeholder - keep existing console logic if needed

    def display_main_menu(self):
        """Display console menu (CONSOLE USE)."""
        # ... (Implementation uses print - suitable for console) ...
        pass # Placeholder - keep existing console logic if needed

    def run_console(self):
        """Main console application loop (CONSOLE USE)."""
        # This method contains the loop calling display_main_menu, getting input,
        # and routing to console methods like get_simple_option_price, manage_favorites etc.
        print("Welcome to Options Analyzer (Console Mode)")
        self._fetch_risk_free_rate_silently() # Ensure rate is loaded

        while True:
             # Example menu structure
             self.clear_screen()
             print("===== Options Analyzer Console =====")
             if self.current_ticker: print(f"Current: {self.current_ticker}")
             print("1. Fetch Stock Data")
             print("2. Simple Option Price")
             print("3. View Options Chain (Console)") # Needs separate console display logic
             print("4. Analyze Strategy (Console)")
             # Add favs/settings options
             print("0. Exit")
             choice = input("Enter choice: ")

             if choice == '1':
                  ticker = input("Enter ticker: ").upper()
                  if ticker: self.get_stock_data(ticker) # Prints details
             elif choice == '2':
                  self.get_simple_option_price() # Uses input/print
             elif choice == '3':
                   # Need specific logic to fetch and print chain using tabulate here
                   print("Fetching chain (console display needs implementation)...")
                   # Example call:
                   # chain_df = self.calculate_options_chain() # Gets data DF
                   # if chain_df is not None:
                   #      print(tabulate(chain_df, headers='keys', tablefmt='psql', showindex=False))
                   pass
             elif choice == '4':
                   self.analyze_strategy() # Uses input/print
             elif choice == '0': break
             else: print("Invalid choice.")
             input("\nPress Enter to continue...")


if __name__ == "__main__":
    # To run in console mode:
    analyzer_console = OptionsAnalyzer()
    analyzer_console.run_console()

    # To use as a backend module (e.g., with app.py),
    # just import the class: from options_analyzer import OptionsAnalyzer
    # The __main__ block above won't run when imported.