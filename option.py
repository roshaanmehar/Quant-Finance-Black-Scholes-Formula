# options_analyzer_corrected.py
# Enhanced Options Pricing and Analysis Tool (Console Version - Corrected)

# Dependencies: numpy, pandas, yfinance, matplotlib, scipy, tabulate
# Install using: pip install numpy pandas yfinance matplotlib scipy tabulate

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from scipy.stats import norm
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import style, ticker as mticker
import os
import json
import warnings
import traceback # For debug mode

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="Parsing dates in DD/MM/YYYY format is deprecated")
# Suppress yfinance download warnings for console clarity (optional)
# warnings.filterwarnings('ignore', message="^\[\*\*\*\*\*\*\*") # Hides download progress

# Set plotting style
style.use('seaborn-v0_8-darkgrid')

# --- Helper function for robust numeric conversion ---
def safe_float(value, default=np.nan):
    """Safely converts a value to float, returning default on error."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

class OptionsAnalyzer:
    """
    A tool for fetching stock data, calculating option prices (Black-Scholes-Merton
    with dividends), analyzing option chains, calculating Greeks, implied volatility,
    and evaluating common option strategies with payoff diagrams via a console interface.
    Corrected version addressing type errors and method signatures.
    """
    def __init__(self):
        """Initialize the Options Analyzer with default parameters"""
        self.current_ticker = None
        self.current_stock_data = None
        self.risk_free_rate = None
        self.config = self._load_config()
        self.favorite_tickers = self._load_favorite_tickers()
        self._chain_cache = {} # Initialize cache for option chains
        # Fetch initial risk-free rate (can be silent or verbose based on need)
        self.get_risk_free_rate(verbose=True) # Set verbose=True for console

    # --- Configuration and Persistence ---

    def _load_config(self):
        """Load configuration from file or use defaults."""
        default_config = {
            'volatility_days': 252,
            'default_risk_free_rate': 0.04,
            'show_greeks_in_chain': True,
            'max_strikes_chain': 20,
            'iv_precision': 0.0001,
            'iv_max_iterations': 100,
            'strategy_price_range': 0.3,
            'debug_mode': False
        }
        config_path = 'options_config.json'
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Merge defaults with loaded config, ensuring all keys exist
                    config = default_config.copy()
                    config.update(config_data)
                    print("Configuration loaded.")
                    return config
            else:
                print("No config file found, using defaults.")
                return default_config.copy()
        except json.JSONDecodeError:
            print(f"Error reading config file '{config_path}'. Using defaults.")
            return default_config.copy()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return default_config.copy()

    def _save_config(self):
        """Save configuration to file."""
        config_path = 'options_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print("Configuration saved successfully.")
        except Exception as e:
            print(f"Error saving configuration to '{config_path}': {e}")

    def _load_favorite_tickers(self):
        """Load favorite tickers from file."""
        favorites_path = 'favorite_tickers.json'
        try:
            if os.path.exists(favorites_path):
                with open(favorites_path, 'r') as f:
                    favorites = json.load(f)
                    if isinstance(favorites, list): # Basic validation
                         print(f"Loaded {len(favorites)} favorite tickers.")
                         return favorites
                    else:
                         print("Error: Favorites file format incorrect. Starting fresh.")
                         return []
            else:
                 print("No favorite tickers file found.")
                 return []
        except json.JSONDecodeError:
             print(f"Error reading favorites file '{favorites_path}'. Starting fresh.")
             return []
        except Exception as e:
            print(f"Error loading favorite tickers: {e}. Starting fresh.")
            return []

    def _save_favorite_tickers(self):
        """Save favorite tickers to file."""
        favorites_path = 'favorite_tickers.json'
        try:
            with open(favorites_path, 'w') as f:
                json.dump(self.favorite_tickers, f, indent=4)
            print("Favorite tickers saved successfully.")
        except Exception as e:
            print(f"Error saving favorite tickers to '{favorites_path}': {e}")

    # --- Utility Functions ---

    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _format_currency(self, value, currency='USD'):
        """Formats a numeric value as currency."""
        numeric_value = safe_float(value, default=None) # Ensure numeric before formatting
        if numeric_value is None:
            return "N/A"
        try:
            symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
            symbol = symbols.get(currency, '')
            return f"{symbol}{numeric_value:,.2f}"
        except (TypeError, ValueError):
             return "N/A"

    def validate_ticker(self, ticker):
        """Validate if the ticker exists using yfinance."""
        if not ticker or not isinstance(ticker, str):
            print("Ticker symbol must be a non-empty string.")
            return False
        ticker = ticker.upper().strip()
        try:
            print(f"\nValidating ticker '{ticker}'...")
            stock = yf.Ticker(ticker)
            # Use history first as it's often more reliable for existence check
            hist = stock.history(period="5d")
            if hist.empty:
                 # Try info as a fallback
                 info = stock.info
                 # Check for empty info or clear indicators of invalidity
                 if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
                      print(f"Ticker '{ticker}' is not valid or no recent data available (checked history and info).")
                      return False
            else:
                 # If history worked, get info for quoteType check
                 info = stock.info

            q_type = info.get('quoteType', 'N/A')
            if q_type not in ['EQUITY', 'ETF', 'INDEX', 'CURRENCY', 'COMMODITY']: # Added more types that might have options
                 print(f"Warning: Ticker '{ticker}' has quote type '{q_type}'. Options might not be available or standard.")

            print(f"Ticker '{ticker}' appears valid ({info.get('shortName', 'N/A')}).")
            return True
        except Exception as e:
            print(f"Error validating ticker '{ticker}': {e}")
            if self.config['debug_mode']:
                traceback.print_exc()
            return False

    def _select_expiration_date(self, expirations):
        """Lists available expiration dates and prompts user selection (Console)."""
        if not expirations:
            print("No expiration dates available for this ticker.")
            return None

        print("\nAvailable expiration dates:")
        valid_expirations = []
        today = dt.datetime.now().date()
        for i, date_str in enumerate(expirations):
            try:
                # Check if date_str is already a date object (less likely from yf)
                if isinstance(date_str, dt.date):
                    exp_date = date_str
                    date_str_fmt = exp_date.strftime('%Y-%m-%d')
                elif isinstance(date_str, str):
                    exp_date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
                    date_str_fmt = date_str
                else:
                    continue # Skip unrecognized format

                days = (exp_date - today).days
                if days >= 0:
                    print(f"{len(valid_expirations) + 1}. {date_str_fmt} ({days} days)")
                    valid_expirations.append({'index': i, 'date': date_str_fmt, 'days': days})
            except ValueError:
                 if self.config['debug_mode']: print(f"Skipping invalid date format: {date_str}")
                 continue

        if not valid_expirations:
            print("No valid future expiration dates found.")
            return None

        while True:
            try:
                selection = input(f"\nSelect expiration date (1-{len(valid_expirations)}), Enter for first: ").strip()
                if not selection:
                    selected_exp = valid_expirations[0]
                    print(f"Using first available date: {selected_exp['date']}")
                    break
                idx = int(selection) - 1
                if 0 <= idx < len(valid_expirations):
                    selected_exp = valid_expirations[idx]
                    break
                else: print("Invalid selection. Please enter a number from the list.")
            except ValueError: print("Invalid input. Please enter a number.")

        print(f"\nSelected expiration date: {selected_exp['date']} ({selected_exp['days']} days)")
        return selected_exp['date']

    # --- Data Fetching ---

    def get_stock_data(self, ticker):
        """Fetch stock data, company info, dividend yield, and options expirations."""
        if not isinstance(ticker, str): return None # Basic type check
        ticker = ticker.upper().strip()
        if not self.validate_ticker(ticker): return None

        try:
            print(f"\nFetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            # Fetch slightly longer history for robustness if needed later
            hist = stock.history(period="1y", interval="1d") # Explicit interval

            current_price_num = None
            volatility_num = None

            # Calculate Volatility from History
            if not hist.empty and 'Close' in hist.columns:
                # Ensure Close is numeric
                hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
                hist = hist.dropna(subset=['Close']) # Drop rows where Close couldn't be converted
                if not hist.empty:
                    current_price_num = safe_float(hist['Close'].iloc[-1]) # Get last valid close
                    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                    if len(returns) >= 10: # Need a reasonable number of returns
                        volatility_num = safe_float(returns.std() * np.sqrt(self.config['volatility_days']))
                    else: print(f"Warning: Only {len(returns)} returns found. Insufficient for reliable volatility calc.")
            else: print(f"Warning: Could not fetch sufficient historical data or 'Close' column missing for {ticker}.")

            # Get Company Info, Fallback Price, and Dividend Yield
            info = {}
            company_name = ticker
            sector, industry, market_cap_str, currency = 'N/A', 'N/A', 'N/A', 'USD'
            dividend_yield_num = 0.0 # Default numeric

            try:
                info = stock.info
                company_name = info.get('shortName', ticker)
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                market_cap = safe_float(info.get('marketCap')) # Ensure numeric
                currency = info.get('currency', 'USD')
                # Fetch dividend yield, ensure numeric
                dividend_yield_num = safe_float(info.get('dividendYield', 0.0), default=0.0)

                # Fallback for current price if history failed or gave NaN
                if current_price_num is None or np.isnan(current_price_num):
                     price_keys = ['regularMarketPrice', 'currentPrice', 'previousClose', 'regularMarketOpen']
                     for key in price_keys:
                         price_val = safe_float(info.get(key))
                         if price_val is not None and not np.isnan(price_val) and price_val > 0:
                             current_price_num = price_val
                             print(f"Using price from info key '{key}': {self._format_currency(current_price_num, currency)}")
                             break
                     # Final fallback: midpoint of day range
                     if current_price_num is None or np.isnan(current_price_num):
                         high = safe_float(info.get('dayHigh'))
                         low = safe_float(info.get('dayLow'))
                         if high is not None and low is not None and not np.isnan(high) and not np.isnan(low) and high >= low > 0:
                             current_price_num = (high + low) / 2
                             print(f"Warning: Using midpoint of day's high/low as current price: {self._format_currency(current_price_num, currency)}")

                # Format Market Cap (only if market_cap is a valid number)
                if market_cap is not None and not np.isnan(market_cap):
                    if market_cap >= 1e12: market_cap_str = f"{self._format_currency(market_cap / 1e12, currency)}T"
                    elif market_cap >= 1e9: market_cap_str = f"{self._format_currency(market_cap / 1e9, currency)}B"
                    elif market_cap >= 1e6: market_cap_str = f"{self._format_currency(market_cap / 1e6, currency)}M"
                    else: market_cap_str = self._format_currency(market_cap, currency)

            except Exception as e_info:
                print(f"Warning: Could not fetch some company info details: {e_info}")
                if self.config['debug_mode']: traceback.print_exc()

            # --- CRITICAL CHECK ---
            if current_price_num is None or np.isnan(current_price_num):
                 raise ValueError(f"CRITICAL: Failed to determine a valid current price for {ticker}.")

            # Print stock details
            print(f"\n=== {company_name} ({ticker}) ===")
            print(f"Current price: {self._format_currency(current_price_num, currency)}")
            print(f"Sector: {sector} | Industry: {industry}")
            print(f"Market Cap: {market_cap_str}")
            print(f"Dividend Yield: {dividend_yield_num:.4f} ({dividend_yield_num*100:.2f}%)")
            if volatility_num is not None and not np.isnan(volatility_num):
                print(f"Annualized Volatility (1y): {volatility_num:.4f} ({volatility_num*100:.2f}%)")
            else: print("Annualized Volatility (1y): N/A")

            # Get available expiration dates
            expirations = ()
            try:
                 expirations = stock.options
                 if not expirations: print("Note: No options expiration dates found.")
            except Exception as e: print(f"Warning: Could not fetch options expiration dates: {e}")

            self.current_ticker = ticker
            self.current_stock_data = {
                'ticker': ticker,
                'current_price': current_price_num, # Store as number
                'volatility': volatility_num,       # Store as number or NaN
                'dividend_yield': dividend_yield_num, # Store as number
                'expirations': expirations,
                'ticker_object': stock,
                'history': hist,
                'info': info,
                'currency': currency
            }
            print(f"\nData fetch complete for {ticker}.")
            return self.current_stock_data

        except Exception as e:
            print(f"\nError fetching data for '{ticker}': {e}")
            if self.config['debug_mode']: traceback.print_exc()
            if self.current_ticker == ticker: self.current_ticker, self.current_stock_data = None, None
            return None

    def get_risk_free_rate(self, verbose=False):
        """Get risk-free rate from Treasury yield (^TNX) or default."""
        try:
            if verbose: print("Fetching current risk-free rate (10-Year Treasury Yield)...")
            treasury = yf.Ticker("^TNX")
            # Use download for potentially more robust fetching
            data = treasury.download(period="5d", progress=False) # progress=False for cleaner console
            if not data.empty and 'Close' in data.columns:
                rate_num = safe_float(data['Close'].iloc[-1]) / 100.0 # Ensure float division
                # Sanity check for the rate
                if rate_num is not None and 0 <= rate_num <= 0.2:
                    if verbose: print(f"Using current risk-free rate (10Y Treasury): {rate_num:.4f} ({rate_num*100:.2f}%)")
                    self.risk_free_rate = rate_num
                    return rate_num
                else:
                     if verbose: print(f"Warning: Fetched treasury rate ({rate_num}) seems unusual or invalid. Falling back.")
            else:
                 if verbose: print("Could not fetch valid treasury data. Falling back to default.")
        except Exception as e:
            if verbose: print(f"Error fetching risk-free rate: {e}. Falling back to default.")
            if self.config['debug_mode']: traceback.print_exc()

        # Fallback to default
        default_rate = safe_float(self.config.get('default_risk_free_rate', 0.04), default=0.04)
        if verbose: print(f"Using default risk-free rate: {default_rate:.4f} ({default_rate*100:.2f}%)")
        self.risk_free_rate = default_rate
        return default_rate

    def _get_option_data_for_strike(self, expiration_date, strike, option_type):
         """Helper to get specific option data (call or put) Series for a strike."""
         if not self.current_stock_data or not self.current_stock_data.get('ticker_object'):
             print("Error: Stock data not loaded.")
             return None
         if not expiration_date: print("Error: Expiration date required."); return None
         if strike <= 0: print("Error: Strike must be positive."); return None

         stock = self.current_stock_data['ticker_object']
         option_type = option_type.lower()
         strike_num = safe_float(strike) # Ensure strike is numeric

         try:
             # Use cache if date matches
             cache_key = f"{expiration_date}"
             if cache_key not in self._chain_cache or self._chain_cache[cache_key]['date'] != expiration_date:
                 if self.config['debug_mode']: print(f"Cache miss or new date. Fetching chain for {expiration_date}...")
                 opt_chain = stock.option_chain(expiration_date)
                 # Store both calls and puts in cache for this date
                 self._chain_cache[cache_key] = {'date': expiration_date, 'calls': opt_chain.calls, 'puts': opt_chain.puts}
             # else:
                 # if self.config['debug_mode']: print(f"Cache hit for {expiration_date}")

             data_df = self._chain_cache[cache_key]['calls'] if option_type == 'call' else self._chain_cache[cache_key]['puts']

             # Find the specific strike row - ensure strike column is numeric first
             if 'strike' not in data_df.columns: return None # Should not happen
             data_df['strike'] = pd.to_numeric(data_df['strike'], errors='coerce')
             option_data_series = data_df[data_df['strike'] == strike_num] # Match numeric strike

             if option_data_series.empty:
                 # print(f"Warning: No {option_type} data found for strike {strike_num} on {expiration_date}.") # Reduced verbosity
                 return None

             return option_data_series.iloc[0] # Return the Series

         except IndexError:
              print(f"Error: No options data structure available for {self.current_ticker} on {expiration_date}.")
              self._chain_cache.pop(cache_key, None) # Clear specific cache entry on error
              return None
         except Exception as e:
             print(f"Error fetching option data for strike {strike_num} ({option_type}): {e}")
             if self.config['debug_mode']: traceback.print_exc()
             self._chain_cache.pop(cache_key, None)
             return None

    # --- Black-Scholes-Merton Model and Greeks (with Dividend Yield q) ---

    def black_scholes_merton(self, S, K, T, r, q, sigma, option_type="call"):
        """ BSM Option Price including continuous dividend yield q. Ensures numeric inputs. """
        # Ensure inputs are numeric
        S, K, T, r, q, sigma = map(safe_float, [S, K, T, r, q, sigma])
        if any(np.isnan(x) for x in [S, K, T, r, q, sigma]): return np.nan

        if T < 0: T = 0
        if sigma <= 0: sigma = 1e-6 # Minimal volatility
        if S <= 0 or K <= 0: return np.nan

        if T == 0: # Handle expiration
            return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)

        try:
            sqrt_T = np.sqrt(T)
            # Check for potential division by zero
            if sigma * sqrt_T == 0: return np.nan

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

            if option_type.lower() == "call":
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == "put":
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            else: return np.nan
            return max(0.0, price)

        except (OverflowError, ValueError) as e: return np.nan
        except Exception as e:
            if self.config['debug_mode']: print(f"BSM Error: {e}"); traceback.print_exc()
            return np.nan

    def calculate_option_greeks(self, S, K, T, r, q, sigma, option_type="call"):
        """ Calculate option Greeks including dividend yield q. Ensures numeric inputs. """
        greeks = { "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan }
        S, K, T, r, q, sigma = map(safe_float, [S, K, T, r, q, sigma])
        if any(np.isnan(x) for x in [S, K, T, r, q, sigma]): return greeks

        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            # Handle greeks at expiration (T=0)
            if T <= 0:
                option_type = option_type.lower()
                delta = np.nan
                if option_type == 'call': delta = 1.0 if S > K else (0.5 if S == K else 0.0)
                elif option_type == 'put': delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
                return { "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0 }
            return greeks # Return NaNs for other invalid inputs (e.g., sigma=0 but T>0)

        option_type = option_type.lower()
        try:
            sqrt_T = np.sqrt(T)
            # Check for potential division by zero
            if sigma * sqrt_T == 0: return {k: 0.0 for k in greeks}

            exp_qT = np.exp(-q * T)
            exp_rT = np.exp(-r * T)

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_neg_d2 = norm.cdf(-d2)

            # Gamma (Ensure denominator is non-zero)
            denom_gamma = S * sigma * sqrt_T
            greeks["gamma"] = (exp_qT * pdf_d1 / denom_gamma) if denom_gamma != 0 else 0

            # Vega
            greeks["vega"] = (S * exp_qT * sqrt_T * pdf_d1) / 100

            # Theta (Ensure sqrt_T is non-zero)
            theta_term1 = - (S * exp_qT * pdf_d1 * sigma) / (2 * sqrt_T) if sqrt_T != 0 else 0

            if option_type == "call":
                greeks["delta"] = exp_qT * cdf_d1
                theta_term2 = - r * K * exp_rT * cdf_d2
                theta_term3 = + q * S * exp_qT * cdf_d1
                greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
                greeks["rho"] = (K * T * exp_rT * cdf_d2) / 100
            elif option_type == "put":
                greeks["delta"] = exp_qT * (cdf_d1 - 1)
                theta_term2 = + r * K * exp_rT * cdf_neg_d2
                theta_term3 = - q * S * exp_qT * cdf_neg_d1
                greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
                greeks["rho"] = (-K * T * exp_rT * cdf_neg_d2) / 100
            else: return {k: np.nan for k in greeks}

            return greeks

        except (ZeroDivisionError, OverflowError, ValueError) as e: return {k: np.nan for k in greeks}
        except Exception as e:
            if self.config['debug_mode']: print(f"Greeks Error: {e}"); traceback.print_exc()
            return {k: np.nan for k in greeks}

    def calculate_implied_volatility(self, S, K, T, r, q, market_price, option_type="call"):
        """ Calculate implied volatility using bisection. Ensures numeric inputs. """
        S, K, T, r, q, market_price = map(safe_float, [S, K, T, r, q, market_price])
        if any(np.isnan(x) for x in [S, K, T, r, q, market_price]): return np.nan

        option_type = option_type.lower()
        precision = self.config['iv_precision']
        max_iterations = self.config['iv_max_iterations']

        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0: return np.nan

        try:
            intrinsic_value = 0.0
            if option_type == "call": intrinsic_value = max(0.0, S * np.exp(-q*T) - K * np.exp(-r * T))
            elif option_type == "put": intrinsic_value = max(0.0, K * np.exp(-r * T) - S * np.exp(-q*T))
            else: return np.nan
        except OverflowError: return np.nan # Cannot calculate intrinsic

        if market_price < intrinsic_value - precision: return np.nan # Below intrinsic

        vol_low, vol_high = 1e-5, 5.0
        price_low = self.black_scholes_merton(S, K, T, r, q, vol_low, option_type)
        price_high = self.black_scholes_merton(S, K, T, r, q, vol_high, option_type)

        if np.isnan(price_low) or np.isnan(price_high): return np.nan # Error at bounds
        if market_price <= price_low: return vol_low
        if market_price >= price_high: return vol_high

        for _ in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            price_mid = self.black_scholes_merton(S, K, T, r, q, vol_mid, option_type)

            if np.isnan(price_mid): return np.nan # BSM failed during iteration

            if abs(price_mid - market_price) < precision: return vol_mid
            if price_mid > market_price: vol_high = vol_mid
            else: vol_low = vol_mid
            if abs(vol_high - vol_low) < precision: break # Converged based on vol range

        # Final check after loop
        final_vol = (vol_low + vol_high) / 2
        final_price = self.black_scholes_merton(S, K, T, r, q, final_vol, option_type)
        if pd.notna(final_price) and abs(final_price - market_price) < precision * 10:
             return final_vol
        else: return np.nan # Failed to converge reliably

    # --- Core Functionality Methods (Console Focused) ---

    def get_simple_option_price(self):
        """Calculate and display a simple option price based on user input (Console)."""
        # --- Ticker Selection/Validation ---
        if self.current_stock_data is None:
            print("\nPlease fetch stock data first (Option 1).")
            ticker_in = input("Enter stock ticker symbol: ").strip()
            if not ticker_in or not self.get_stock_data(ticker_in): return
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change = input("Fetch data for a different ticker? (y/n, default n): ").lower().strip()
             if change == 'y':
                  ticker_in = input("Enter new stock ticker symbol: ").strip()
                  if not ticker_in or not self.get_stock_data(ticker_in): return

        stock_data = self.current_stock_data
        if not stock_data: print("Error: Stock data not available."); return

        # --- Get Data (ensure numeric) ---
        current_price = stock_data['current_price'] # Should be numeric from get_stock_data
        volatility = stock_data['volatility']       # Numeric or NaN
        dividend_yield = stock_data['dividend_yield'] # Numeric
        expirations = stock_data['expirations']
        currency = stock_data['currency']
        risk_free_rate = self.risk_free_rate           # Should be numeric

        # Handle missing/NaN volatility
        vol_to_use = volatility
        if vol_to_use is None or np.isnan(vol_to_use):
             print("\nWarning: Historical volatility unavailable/NaN.")
             while True:
                 try:
                     user_vol_str = input("Enter estimated annual volatility (e.g., 0.3) or Enter for 0.3: ").strip()
                     vol_to_use = safe_float(user_vol_str, default=0.3) # Default 30%
                     if vol_to_use > 0: break
                     elif user_vol_str == "": vol_to_use = 0.3; break # Handle enter press
                     else: print("Volatility must be positive.")
                 except ValueError: print("Invalid input. Please enter a number.")
             print(f"Using estimated volatility: {vol_to_use*100:.2f}%")

        if risk_free_rate is None: risk_free_rate = self.get_risk_free_rate(verbose=True)

        # --- Select Expiration ---
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date: return

        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days)
        T_years = days_to_expiration / 365.0
        print(f"Time to expiration (T): {T_years:.4f} years ({days_to_expiration} days)")

        # --- Select Strike Price (ensure numeric) ---
        strike_num = None
        while strike_num is None:
             strike_input = input(f"\nEnter strike price (e.g., {current_price:.2f}) or 'atm': ").lower().strip()
             if strike_input == 'atm':
                 try:
                    options = stock_data['ticker_object'].option_chain(expiration_date) # Use stored object
                    all_strikes_series = pd.concat([options.calls['strike'], options.puts['strike']]).dropna().unique()
                    all_strikes = sorted([s for s in all_strikes_series if safe_float(s) is not None]) # Ensure numeric strikes
                    if not all_strikes:
                        print("No valid strikes found. Using current price.")
                        strike_num = current_price
                    else:
                        strike_num = min(all_strikes, key=lambda x: abs(x - current_price))
                        print(f"Found closest available strike: {self._format_currency(strike_num, currency)}")
                 except Exception as e:
                    print(f"Could not fetch/process strikes for ATM. Using current price. Error: {e}")
                    strike_num = current_price # Fallback
             else:
                 strike_val = safe_float(strike_input)
                 if strike_val is not None and strike_val > 0:
                     strike_num = strike_val
                 else: print("Invalid input. Strike must be a positive number or 'atm'.")

        # --- Select Option Type ---
        option_type_sel = None
        while option_type_sel not in ['call', 'put', 'both']:
            opt_in = input("Calculate for 'call', 'put', or 'both'? (both): ").lower().strip() or 'both'
            if opt_in in ['call', 'put', 'both']: option_type_sel = opt_in
            else: print("Invalid option type.")

        # --- Calculate and Display ---
        print(f"\n--- BSM Option Analysis ---")
        print(f"Stock: {self.current_ticker} @ {self._format_currency(current_price, currency)}")
        print(f"Strike: {self._format_currency(strike_num, currency)}")
        print(f"Expiration: {expiration_date} ({days_to_expiration} days, T={T_years:.4f})")
        print(f"Volatility (Input): {vol_to_use*100:.2f}%")
        print(f"Risk-Free Rate: {risk_free_rate*100:.2f}%")
        print(f"Dividend Yield: {dividend_yield*100:.2f}%")
        print("-" * 30)

        if option_type_sel in ['call', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike_num, T_years, risk_free_rate, dividend_yield, vol_to_use, "call")
            greeks = self.calculate_option_greeks(current_price, strike_num, T_years, risk_free_rate, dividend_yield, vol_to_use, "call")
            print(f"BSM Call Price: {self._format_currency(bsm_price, currency)}")
            if greeks and not all(np.isnan(v) for v in greeks.values()):
                print("  Greeks (Call):")
                print(f"    Delta: {greeks['delta']:.4f}")
                print(f"    Gamma: {greeks['gamma']:.4f}")
                print(f"    Theta: {self._format_currency(greeks['theta'], currency)} / day")
                print(f"    Vega:  {self._format_currency(greeks['vega'], currency)} / 1% vol")
                print(f"    Rho:   {self._format_currency(greeks['rho'], currency)} / 1% rate")
            else: print("  Greeks (Call): Calculation failed or N/A")
            print("-" * 30)

        if option_type_sel in ['put', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike_num, T_years, risk_free_rate, dividend_yield, vol_to_use, "put")
            greeks = self.calculate_option_greeks(current_price, strike_num, T_years, risk_free_rate, dividend_yield, vol_to_use, "put")
            print(f"BSM Put Price: {self._format_currency(bsm_price, currency)}")
            if greeks and not all(np.isnan(v) for v in greeks.values()):
                print("  Greeks (Put):")
                print(f"    Delta: {greeks['delta']:.4f}")
                print(f"    Gamma: {greeks['gamma']:.4f}")
                print(f"    Theta: {self._format_currency(greeks['theta'], currency)} / day")
                print(f"    Vega:  {self._format_currency(greeks['vega'], currency)} / 1% vol")
                print(f"    Rho:   {self._format_currency(greeks['rho'], currency)} / 1% rate")
            else: print("  Greeks (Put): Calculation failed or N/A")
            print("-" * 30)

    def calculate_options_chain(self, ticker=None, specific_expiration=None, visualize=False):
        """
        Calculate and display/return a detailed options chain.
        Accepts optional ticker and expiration for non-console use.
        """
        # --- Determine Ticker and Fetch Data ---
        target_ticker = ticker
        if target_ticker is None: # Use console logic if ticker not provided
            if self.current_stock_data is None:
                 print("\nPlease fetch stock data first (Option 1).")
                 ticker_in = input("Enter stock ticker symbol: ").strip().upper()
                 if not ticker_in: return None
                 target_ticker = ticker_in
            else:
                 print(f"\nCurrent ticker: {self.current_ticker}")
                 change = input("Fetch data for a different ticker? (y/n, default n): ").lower().strip()
                 if change == 'y':
                      ticker_in = input("Enter new stock ticker symbol: ").strip().upper()
                      if not ticker_in: return None
                      target_ticker = ticker_in
                 else:
                      target_ticker = self.current_ticker
        else:
             target_ticker = target_ticker.strip().upper() # Use provided ticker

        # Fetch or use existing data
        if self.current_stock_data is None or self.current_ticker != target_ticker:
            if not self.get_stock_data(target_ticker): return None
        stock_data = self.current_stock_data
        if not stock_data: print("Error: Stock data could not be loaded."); return None

        # --- Get Numeric Data ---
        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        dividend_yield = stock_data['dividend_yield']
        expirations = stock_data.get('expirations', ())
        stock = stock_data['ticker_object'] # yf.Ticker object
        currency = stock_data.get('currency', 'USD')
        risk_free_rate = self.risk_free_rate

        # --- Determine Expiration Date ---
        expiration_date = specific_expiration
        if expiration_date is None: # Use console logic if expiration not provided
            expiration_date = self._select_expiration_date(expirations)
            if not expiration_date: return None
        elif expirations and expiration_date not in expirations:
             print(f"Error: Provided expiration '{expiration_date}' not found in available dates: {expirations}")
             return None
        elif not expirations:
             print(f"Error: No expirations found for {target_ticker}.")
             return None

        # --- Handle Missing Volatility (for console case mostly) ---
        volatility_input = volatility # Store original/fetched
        if volatility is None or np.isnan(volatility):
             if specific_expiration is None: # Only prompt if running via console menu path
                 print("\nWarning: Historical volatility unavailable/NaN.")
                 while True:
                     try:
                         user_vol_str = input("Enter est. annual vol (e.g., 0.3) or Enter for 0.3: ").strip()
                         volatility = safe_float(user_vol_str, default=0.3)
                         if volatility > 0: break
                         elif user_vol_str == "": volatility = 0.3; break
                         else: print("Volatility must be positive.")
                     except ValueError: print("Invalid input.")
                 print(f"Using estimated volatility for BSM/Greeks: {volatility*100:.2f}%")
             else: # If called programmatically, use a default or signal error? Use 0.3 default for now.
                 print(f"Warning: Historical volatility unavailable/NaN for {target_ticker}. Using default 0.3 for BSM estimates.")
                 volatility = 0.3

        if risk_free_rate is None: risk_free_rate = self.get_risk_free_rate(verbose=True)

        # --- Calculate Time to Expiration ---
        try:
            today = dt.datetime.now().date()
            exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
            days_to_expiration = max(0, (exp_date - today).days)
            T = days_to_expiration / 365.0
            print(f"Time to expiration (T): {T:.4f} years ({days_to_expiration} days)")
        except ValueError:
             print(f"Error: Invalid expiration date format '{expiration_date}'. Expected YYYY-MM-DD.")
             return None

        # --- Fetch Option Chain Data ---
        try:
            print(f"\nFetching options chain data for {target_ticker} ({expiration_date})...")
            self._chain_cache = {} # Clear cache for new request
            options = stock.option_chain(expiration_date)
            calls_df = options.calls
            puts_df = options.puts
            self._chain_cache[expiration_date] = {'date': expiration_date, 'calls': calls_df, 'puts': puts_df}

            if calls_df.empty and puts_df.empty:
                 print(f"No options data found for {target_ticker} on {expiration_date}.")
                 return pd.DataFrame() # Return empty DataFrame

            print("Processing chain: Calculating BSM prices, IV, and Greeks...")

            # Prepare DataFrames
            calls_df = calls_df.add_prefix('call_').rename(columns={'call_strike': 'strike'})
            puts_df = puts_df.add_prefix('put_').rename(columns={'put_strike': 'strike'})

            # --- Ensure key columns are numeric ---
            num_cols_c = ['strike', 'call_lastPrice', 'call_bid', 'call_ask', 'call_volume', 'call_openInterest', 'call_impliedVolatility']
            num_cols_p = ['strike', 'put_lastPrice', 'put_bid', 'put_ask', 'put_volume', 'put_openInterest', 'put_impliedVolatility']

            for col in num_cols_c:
                if col in calls_df.columns: calls_df[col] = pd.to_numeric(calls_df[col], errors='coerce')
            for col in num_cols_p:
                if col in puts_df.columns: puts_df[col] = pd.to_numeric(puts_df[col], errors='coerce')

            # Select relevant columns AFTER ensuring numeric types
            rel_cols_c = [c for c in num_cols_c if c in calls_df.columns]
            rel_cols_p = [p for p in num_cols_p if p in puts_df.columns]

            # Merge on numeric strike
            chain_df = pd.merge(calls_df[rel_cols_c], puts_df[rel_cols_p], on='strike', how='outer')
            chain_df = chain_df.sort_values(by='strike').reset_index(drop=True)
            chain_df = chain_df.dropna(subset=['strike']) # Drop rows where strike itself is NaN

            # --- Limit Strikes Around ATM ---
            max_strikes = self.config['max_strikes_chain']
            if len(chain_df) > max_strikes and current_price is not None:
                # Ensure current_price is valid for comparison
                chain_df['strike_diff'] = (chain_df['strike'] - current_price).abs()
                atm_index = chain_df['strike_diff'].idxmin() # Find index of minimum difference
                atm_pos = chain_df.index.get_loc(atm_index) # Get position of that index

                half_width = max_strikes // 2
                start_idx = max(0, atm_pos - half_width)
                end_idx = min(len(chain_df), start_idx + max_strikes)
                if (end_idx - start_idx) < max_strikes: start_idx = max(0, end_idx - max_strikes)
                chain_df = chain_df.iloc[start_idx:end_idx].reset_index(drop=True)
                print(f"Displaying {len(chain_df)} strikes around current price {self._format_currency(current_price, currency)}")

            # --- Calculate BSM, IV, Greeks (Iterate through DataFrame) ---
            results_list = [] # Use list of dicts for efficiency
            total_strikes = len(chain_df)
            for idx, row in chain_df.iterrows():
                print(f"\rProcessing strike {idx+1}/{total_strikes}...", end="")

                # Get strike (already numeric)
                strike_k = row['strike']
                data_row = {'strike': strike_k}

                # Get market prices (already numeric or NaN)
                call_bid, call_ask, market_call = row.get('call_bid'), row.get('call_ask'), row.get('call_lastPrice')
                put_bid, put_ask, market_put = row.get('put_bid'), row.get('put_ask'), row.get('put_lastPrice')

                # Use midpoint if bid/ask are valid and last price is suspect or zero
                market_call_use = market_call
                if call_bid > 0 and call_ask > call_bid: # Valid bid/ask spread
                    mid_c = (call_bid + call_ask) / 2
                    # Use mid if last is zero, NaN, or outside bid/ask
                    if pd.isna(market_call) or market_call <=0 or market_call < call_bid or market_call > call_ask:
                        market_call_use = mid_c
                    # Optional: Use mid even if last is valid but far from mid (more debatable)
                    # elif abs(market_call - mid_c) > mid_c * 0.2: # If last is > 20% away from mid
                    #     market_call_use = mid_c

                market_put_use = market_put
                if put_bid > 0 and put_ask > put_bid: # Valid bid/ask spread
                    mid_p = (put_bid + put_ask) / 2
                    if pd.isna(market_put) or market_put <=0 or market_put < put_bid or market_put > put_ask:
                        market_put_use = mid_p

                data_row['market_call'] = market_call_use
                data_row['market_put'] = market_put_use

                # Calculate IV using the price we decided to use
                call_iv_calc = self.calculate_implied_volatility(current_price, strike_k, T, risk_free_rate, dividend_yield, market_call_use, "call")
                put_iv_calc = self.calculate_implied_volatility(current_price, strike_k, T, risk_free_rate, dividend_yield, market_put_use, "put")
                data_row['call_iv'] = call_iv_calc * 100 if pd.notna(call_iv_calc) else np.nan
                data_row['put_iv'] = put_iv_calc * 100 if pd.notna(put_iv_calc) else np.nan
                data_row['call_iv_yf'] = row.get('call_impliedVolatility', np.nan) * 100 # YF IV
                data_row['put_iv_yf'] = row.get('put_impliedVolatility', np.nan) * 100  # YF IV

                # Determine volatility for BSM/Greeks
                yf_iv_c = row.get('call_impliedVolatility')
                vol_c = call_iv_calc if pd.notna(call_iv_calc) else (yf_iv_c if pd.notna(yf_iv_c) else volatility)
                yf_iv_p = row.get('put_impliedVolatility')
                vol_p = put_iv_calc if pd.notna(put_iv_calc) else (yf_iv_p if pd.notna(yf_iv_p) else volatility)

                # Calculate BSM Price and Greeks
                data_row['bsm_call'] = self.black_scholes_merton(current_price, strike_k, T, risk_free_rate, dividend_yield, vol_c, "call")
                data_row['bsm_put'] = self.black_scholes_merton(current_price, strike_k, T, risk_free_rate, dividend_yield, vol_p, "put")

                if self.config['show_greeks_in_chain']:
                     greeks_c = self.calculate_option_greeks(current_price, strike_k, T, risk_free_rate, dividend_yield, vol_c, "call")
                     greeks_p = self.calculate_option_greeks(current_price, strike_k, T, risk_free_rate, dividend_yield, vol_p, "put")
                     data_row.update({f'call_{k}': v for k, v in greeks_c.items()})
                     data_row.update({f'put_{k}': v for k, v in greeks_p.items()})

                # Other info (already numeric or NaN)
                data_row['call_volume'] = row.get('call_volume')
                data_row['call_oi'] = row.get('call_openInterest')
                data_row['put_volume'] = row.get('put_volume')
                data_row['put_oi'] = row.get('put_openInterest')

                results_list.append(data_row)

            print("\r" + " " * 50 + "\rProcessing complete.") # Clear progress line

            if not results_list: return pd.DataFrame() # Return empty if loop didn't run

            results_df = pd.DataFrame(results_list) # Create final DataFrame

            # --- Display Results Table (Console) ---
            # Define columns based on config and calculated data
            col_map = {
                'call_volume': 'C Vol', 'call_oi': 'C OI', 'market_call': 'C Market', 'bsm_call': 'C BSM', 'call_iv': 'C IV%', 'call_iv_yf': 'C IV%(YF)',
                'call_delta': 'C Delta', 'call_gamma': 'C Gamma', 'call_theta': 'C Theta', 'call_vega': 'C Vega', 'call_rho': 'C Rho',
                'strike': 'Strike',
                'put_rho': 'P Rho', 'put_vega': 'P Vega', 'put_theta': 'P Theta', 'put_gamma': 'P Gamma', 'put_delta': 'P Delta',
                'put_iv': 'P IV%', 'put_iv_yf': 'P IV%(YF)', 'bsm_put': 'P BSM', 'market_put': 'P Market', 'put_oi': 'P OI', 'put_volume': 'P Vol'
            }
            base_cols_c = ['call_volume', 'call_oi', 'market_call', 'bsm_call', 'call_iv']
            base_cols_p = ['put_iv', 'bsm_put', 'market_put', 'put_oi', 'put_volume']
            greek_cols_c = ['call_delta', 'call_gamma', 'call_theta', 'call_vega', 'call_rho']
            greek_cols_p = ['put_rho', 'put_vega', 'put_theta', 'put_gamma', 'put_delta']

            display_order = base_cols_c
            if self.config['show_greeks_in_chain']: display_order += greek_cols_c
            display_order += ['strike']
            if self.config['show_greeks_in_chain']: display_order += greek_cols_p
            display_order += base_cols_p

            # Filter results_df to only include columns that exist and are in display_order
            display_df = results_df[[col for col in display_order if col in results_df.columns]].copy()
            display_df.rename(columns=col_map, inplace=True) # Rename using the map

            # Apply formatting for display
            formatted_df = display_df.copy()
            for col_disp_name in formatted_df.columns:
                 # Determine original column name for formatting logic if needed
                 # col_orig_name = next((orig for orig, disp in col_map.items() if disp == col_disp_name), None)
                 # Apply formatting based on display name patterns
                 if 'Market' in col_disp_name or 'BSM' in col_disp_name or 'Strike' in col_disp_name or 'Theta' in col_disp_name or 'Vega' in col_disp_name or 'Rho' in col_disp_name:
                     formatted_df[col_disp_name] = formatted_df[col_disp_name].apply(lambda x: self._format_currency(x, currency))
                 elif 'IV%' in col_disp_name:
                     formatted_df[col_disp_name] = formatted_df[col_disp_name].apply(lambda x: f"{safe_float(x):.2f}%" if pd.notna(x) else 'N/A')
                 elif 'Delta' in col_disp_name or 'Gamma' in col_disp_name:
                      formatted_df[col_disp_name] = formatted_df[col_disp_name].apply(lambda x: f"{safe_float(x):.4f}" if pd.notna(x) else 'N/A')
                 elif 'Vol' in col_disp_name or 'OI' in col_disp_name:
                     formatted_df[col_disp_name] = formatted_df[col_disp_name].apply(lambda x: f"{int(safe_float(x, 0)):,}" if pd.notna(x) else '0') # Default 0 if NaN, format as int

            # --- Print Table ---
            print(f"\n--- Options Chain for {self.current_ticker} ---")
            print(f"Expiration: {expiration_date} ({days_to_expiration} days)")
            print(f"Current Price: {self._format_currency(current_price, currency)}")
            print(f"Risk-Free Rate: {risk_free_rate*100:.2f}% | Div Yield: {dividend_yield*100:.2f}%")
            vol_source_disp = "Hist/Est" if volatility_input is None or np.isnan(volatility_input) else "Hist"
            print(f"BSM Volatility ({vol_source_disp}): {volatility*100:.2f}% (used if IV unavailable)")
            print("-" * max(100, len(tabulate(formatted_df.head(1), headers='keys', tablefmt='pretty')))) # Adjust width dynamically?

            print(tabulate(formatted_df, headers='keys', tablefmt='pretty', showindex=False, numalign="right", stralign="right"))
            print("-" * max(100, len(tabulate(formatted_df.head(1), headers='keys', tablefmt='pretty'))))

            # --- Visualize ---
            # Use the passed 'visualize' flag OR prompt if called from console menu
            should_visualize = visualize
            if specific_expiration is None and not visualize: # Check if called from console menu path
                viz_input = input("\nVisualize this options chain (Price/IV)? (y/n): ").lower().strip()
                if viz_input == 'y': should_visualize = True

            if should_visualize:
                 print("Generating visualization...")
                 self.visualize_options_chain(results_df, current_price, currency, expiration_date) # Pass raw results_df

            return results_df # Return the raw calculated data

        except AttributeError as ae:
             print(f"\nAttribute Error: Options data structure likely missing for {target_ticker} on {expiration_date}. {ae}")
             if self.config['debug_mode']: traceback.print_exc()
             return None
        except Exception as e:
            print(f"\nError calculating options chain: {e}")
            if self.config['debug_mode']: traceback.print_exc()
            return None

    def visualize_options_chain(self, df, current_price, currency, expiration_date):
        """Visualize the options chain data. Uses matplotlib."""
        # --- Input Validation ---
        if df is None or df.empty: print("No data to visualize."); return
        required_cols = ['strike']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: DataFrame missing required columns for visualization (needs: {required_cols}). Has: {df.columns.tolist()}")
            return
        # Ensure current price is valid
        current_price_num = safe_float(current_price)
        if current_price_num is None or np.isnan(current_price_num):
            print("Error: Invalid current price provided for visualization.")
            return

        df_vis = df.copy()
        # Ensure strike is numeric for plotting
        df_vis['strike'] = pd.to_numeric(df_vis['strike'], errors='coerce')
        df_vis = df_vis.dropna(subset=['strike'])
        if df_vis.empty: print("No valid strike data to visualize after cleaning."); return

        # --- Plotting Setup ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"{self.current_ticker} Options Chain ({expiration_date})\nCurrent Price: {self._format_currency(current_price_num, currency)}", fontsize=15, weight='bold')
        fig.patch.set_facecolor('white')

        # --- Plot 1: Prices ---
        ax1 = axes[0]; ax1.set_facecolor('#f0f0f0')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{self._format_currency(1, currency)[0]}%.2f'))

        plot_cols_p1 = {
            'market_call': {'label': 'Market Call', 'style': 'bo-', 'alpha': 0.8, 'size': 5},
            'bsm_call':    {'label': 'BSM Call',    'style': 'c--', 'alpha': 0.8, 'size': None},
            'market_put':  {'label': 'Market Put',  'style': 'ro-', 'alpha': 0.8, 'size': 5},
            'bsm_put':     {'label': 'BSM Put',     'style': 'm--', 'alpha': 0.8, 'size': None},
        }
        for col, params in plot_cols_p1.items():
            if col in df_vis.columns:
                 # Ensure data is numeric before plotting
                 data_series = pd.to_numeric(df_vis[col], errors='coerce')
                 if data_series.notna().any():
                     ax1.plot(df_vis['strike'], data_series, params['style'], label=params['label'], markersize=params['size'], alpha=params['alpha'])

        ax1.set_ylabel(f'Option Price ({currency})', fontsize=11, weight='medium')
        ax1.set_title('Market vs. Calculated (BSM) Prices', fontsize=13, weight='medium')
        ax1.grid(True, linestyle=':', linewidth=0.5, color='grey')
        ax1.axvline(current_price_num, color='black', linestyle=':', lw=1.5, label=f'Current Price')
        ax1.legend(fontsize=9, loc='best'); ax1.tick_params(axis='both', which='major', labelsize=10)

        # --- Plot 2: Implied Volatility ---
        ax2 = axes[1]; ax2.set_facecolor('#f0f0f0')
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

        plot_cols_p2 = {
             'call_iv':    {'label': 'Call IV (Calc)', 'style': 'g^-', 'alpha': 0.8, 'size': 5},
             'put_iv':     {'label': 'Put IV (Calc)',  'style': 'yv-', 'alpha': 0.8, 'size': 5}, # Changed color
             'call_iv_yf': {'label': 'Call IV (YF)', 'style': 'c:', 'alpha': 0.7, 'size': 4}, # Changed color
             'put_iv_yf':  {'label': 'Put IV (YF)',  'style': 'm:', 'alpha': 0.7, 'size': 4}, # Changed color
        }
        valid_ivs_plot = []
        for col, params in plot_cols_p2.items():
            if col in df_vis.columns:
                 data_series = pd.to_numeric(df_vis[col], errors='coerce')
                 if data_series.notna().any():
                     ax2.plot(df_vis['strike'], data_series, params['style'], label=params['label'], markersize=params['size'], alpha=params['alpha'])
                     valid_ivs_plot.extend(data_series.dropna().tolist()) # Collect valid IVs for ylim

        # Add historical volatility line
        hist_vol = self.current_stock_data.get('volatility')
        if hist_vol is not None and not np.isnan(hist_vol):
             hist_vol_pct = hist_vol * 100
             ax2.axhline(hist_vol_pct, color='dimgray', linestyle='--', lw=1.5, label=f'Hist. Vol ({hist_vol_pct:.1f}%)')
             valid_ivs_plot.append(hist_vol_pct) # Include for ylim calculation

        ax2.set_xlabel('Strike Price', fontsize=11, weight='medium')
        ax2.set_ylabel('Implied Volatility (%)', fontsize=11, weight='medium')
        ax2.set_title('Implied Volatility Smile / Skew', fontsize=13, weight='medium')
        ax2.grid(True, linestyle=':', linewidth=0.5, color='grey')
        ax2.axvline(current_price_num, color='black', linestyle=':', lw=1.5)
        ax2.legend(fontsize=9, loc='best'); ax2.tick_params(axis='both', which='major', labelsize=10)

        # Adjust y-axis limits for IV plot
        if valid_ivs_plot:
            min_iv, max_iv = min(valid_ivs_plot), max(valid_ivs_plot)
            padding = max((max_iv - min_iv) * 0.1, 5.0) # Add padding, at least 5% IV points
            ax2.set_ylim(max(0, min_iv - padding), max_iv + padding) # Ensure lower limit >= 0

        # --- Display Plot ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try: plt.show()
        except Exception as e: print(f"\nError displaying plot: {e}. Ensure GUI backend.")

    # --- Options Strategy Analysis ---

    def _calculate_payoff(self, S_T, strategy_legs):
        """Calculates the Profit/Loss of a strategy at expiration price S_T."""
        S_T_num = safe_float(S_T) # Ensure numeric
        if np.isnan(S_T_num): return np.nan

        total_payoff = 0
        net_cost = 0
        for leg in strategy_legs:
            # Ensure price and K are numeric
            price = safe_float(leg.get('price'), 0.0) # Default 0 if price missing/invalid
            K = safe_float(leg.get('K')) # Will be NaN if not present or invalid
            direction = leg.get('dir', 'long')
            leg_type = leg.get('type')

            # Accumulate net cost (debit is positive, credit is negative)
            if direction == 'long': net_cost += price
            else: net_cost -= price

            # Calculate terminal value of the leg
            payoff_leg = 0
            if leg_type == 'stock':
                payoff_leg = S_T_num
            elif leg_type == 'call' and not np.isnan(K):
                payoff_leg = max(0.0, S_T_num - K)
            elif leg_type == 'put' and not np.isnan(K):
                payoff_leg = max(0.0, K - S_T_num)

            if direction == 'short': payoff_leg *= -1
            total_payoff += payoff_leg

        profit_loss = total_payoff - net_cost
        return profit_loss

    def _plot_payoff(self, S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency):
        """Plots the Profit/Loss diagram for a strategy."""
        # --- Input Validation ---
        if not isinstance(S_T_range, (np.ndarray, list)) or not isinstance(PnL, (np.ndarray, list)): return
        if len(S_T_range) != len(PnL): return
        max_profit_num = safe_float(max_profit, default=np.inf) # Default to inf if invalid
        max_loss_num = safe_float(max_loss, default=-np.inf) # Default to -inf if invalid

        # --- Plotting Setup ---
        fig, ax = plt.subplots(figsize=(11, 6.5)); fig.patch.set_facecolor('white')
        ax.set_facecolor('#f0f0f0')

        # --- Plot PnL ---
        ax.plot(S_T_range, PnL, lw=2.5, color='navy', label='Profit/Loss at Expiration')
        ax.axhline(0, color='black', linestyle='--', lw=1, label='Breakeven Level')

        # --- Format Axes ---
        curr_symbol = self._format_currency(1, currency)[0] if currency else '$'
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{curr_symbol}%.2f'))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'{curr_symbol}%.2f'))
        ax.tick_params(axis='both', which='major', labelsize=10)

        # --- Annotate Breakevens ---
        valid_bes = sorted([safe_float(be) for be in breakevens if pd.notna(safe_float(be))])
        if valid_bes:
            ax.scatter(valid_bes, [0] * len(valid_bes), color='red', s=100, zorder=5, edgecolors='black', label='Breakeven(s)')
            # Determine dynamic offset for text based on PnL range near BE points
            y_range = max(abs(PnL.min()), abs(PnL.max())) if len(PnL)>0 else 1
            offset = y_range * 0.05 if y_range != 0 else 0.1 # 5% offset or fixed if flat
            for be in valid_bes:
                 ax.text(be, offset, f' BE: {self._format_currency(be, currency)}', color='darkred', ha='center', va='bottom', fontsize=9, weight='bold')

        # --- Annotate Max Profit/Loss ---
        profit_label = f'Max Profit: {self._format_currency(max_profit_num, currency)}' if np.isfinite(max_profit_num) else 'Max Profit: Unlimited'
        loss_label = f'Max Loss: {self._format_currency(max_loss_num, currency)}' if np.isfinite(max_loss_num) else 'Max Loss: Unlimited' # Adjust if needed (e.g., covered call)

        if np.isfinite(max_profit_num):
             ax.axhline(max_profit_num, color='green', linestyle=':', lw=1.5, label=profit_label)
        elif len(S_T_range)>0 and len(PnL)>0: # Indicate unlimited only if data exists
             ax.text(S_T_range[-1], PnL[-1], ' Unlimited Profit ->', color='darkgreen', ha='right', va='center', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.5, pad=0.1))

        if np.isfinite(max_loss_num):
             ax.axhline(max_loss_num, color='red', linestyle=':', lw=1.5, label=loss_label)
        elif len(S_T_range)>0 and len(PnL)>0: # Indicate unlimited only if data exists
             ax.text(S_T_range[0], PnL[0], '<- Unlimited Loss ', color='darkred', ha='left', va='center', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.5, pad=0.1))


        # --- Titles and Labels ---
        ax.set_title(f'{strategy_name} Payoff Diagram', fontsize=15, weight='bold')
        ax.set_xlabel(f'Underlying Price at Expiration ({currency})', fontsize=11, weight='medium')
        ax.set_ylabel(f'Profit / Loss ({currency})', fontsize=11, weight='medium')
        ax.grid(True, linestyle=':', linewidth=0.5, color='grey')

        # --- Y-axis Limits ---
        y_min_plot, y_max_plot = np.nanmin(PnL) if len(PnL)>0 else -1, np.nanmax(PnL) if len(PnL)>0 else 1
        if np.isfinite(max_loss_num): y_min_plot = min(y_min_plot, max_loss_num)
        if np.isfinite(max_profit_num): y_max_plot = max(y_max_plot, max_profit_num)
        padding = (y_max_plot - y_min_plot) * 0.1 if (y_max_plot != y_min_plot) else 1.0 # Add padding
        ax.set_ylim(y_min_plot - padding, y_max_plot + padding)

        # --- Legend and Display ---
        ax.legend(fontsize=9, loc='best'); plt.tight_layout()
        try: plt.show()
        except Exception as e: print(f"\nError displaying plot: {e}.")

    def analyze_strategy(self):
        """Guides user through selecting and analyzing an options strategy (Console)."""
        # --- Get Stock Data ---
        if self.current_stock_data is None:
             print("\nPlease fetch stock data first (Option 1).")
             ticker_in = input("Enter stock ticker symbol: ").strip().upper()
             if not ticker_in or not self.get_stock_data(ticker_in): return
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change = input("Analyze strategy for a different ticker? (y/n, default n): ").lower().strip()
             if change == 'y':
                  ticker_in = input("Enter new stock ticker symbol: ").strip().upper()
                  if not ticker_in or not self.get_stock_data(ticker_in): return

        stock_data = self.current_stock_data
        if not stock_data: print("Error: Stock data unavailable."); return

        S0 = stock_data['current_price']
        expirations = stock_data['expirations']
        currency = stock_data['currency']
        dividend_yield = stock_data['dividend_yield']
        risk_free_rate = self.risk_free_rate
        volatility = stock_data['volatility'] # Base vol for BSM estimate

        # --- Strategy Selection ---
        print("\n--- Options Strategy Analysis ---"); print("Select a strategy:")
        strategies = { 1: "Covered Call", 2: "Protective Put", 3: "Bull Call Spread",
                       4: "Bear Put Spread", 5: "Long Straddle", 6: "Long Strangle"}
        for i, name in strategies.items(): print(f" {i}. {name}")
        print(" 0. Back to Main Menu")
        strategy_choice = None
        while strategy_choice is None:
            try:
                choice_str = input("Enter strategy number: ").strip()
                choice_int = int(choice_str)
                if choice_int == 0: return
                if choice_int in strategies: strategy_choice = choice_int; break
                else: print("Invalid choice.")
            except ValueError: print("Invalid input. Please enter a number.")

        # --- Select Expiration ---
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date: return

        # --- Helper: Get Numeric Option Price (Market or BSM Estimate) ---
        def get_numeric_option_price(strike_p, option_type_p, exp_date_p):
            strike_p_num = safe_float(strike_p)
            if np.isnan(strike_p_num) or strike_p_num <= 0:
                raise ValueError(f"Invalid strike price provided: {strike_p}")

            option_data_series = self._get_option_data_for_strike(exp_date_p, strike_p_num, option_type_p)
            market_price_num = np.nan

            if option_data_series is not None:
                # Try midpoint using safe_float
                bid = safe_float(option_data_series.get('bid'), default=0.0)
                ask = safe_float(option_data_series.get('ask'), default=0.0)
                last = safe_float(option_data_series.get('lastPrice')) # Keep NaN if invalid

                if bid > 0 and ask > bid: # Valid spread
                    mid = (bid + ask) / 2
                    # Use mid if last is invalid or outside bid/ask
                    if np.isnan(last) or last <= 0 or last < bid or last > ask:
                        market_price_num = mid
                        print(f" Note: Using bid/ask midpoint ({self._format_currency(mid, currency)}) for {option_type_p.capitalize()} K={strike_p_num}.")
                    else: # Last price seems valid and within spread
                        market_price_num = last
                elif not np.isnan(last) and last > 0: # Fallback to last if valid and no good spread
                    market_price_num = last
                    print(f" Note: Using last price ({self._format_currency(last, currency)}) for {option_type_p.capitalize()} K={strike_p_num} (bid/ask invalid).")

            # If market price still NaN or zero, use BSM estimate
            if np.isnan(market_price_num) or market_price_num <= 0:
                print(f"Warning: Market price for {option_type_p.capitalize()} K={strike_p_num} unavailable/zero. Using BSM estimate.")
                vol_est = volatility if pd.notna(volatility) else 0.3
                r_est = risk_free_rate if pd.notna(risk_free_rate) else self.config['default_risk_free_rate']
                q_est = dividend_yield
                today = dt.datetime.now().date()
                exp_d_dt = dt.datetime.strptime(exp_date_p, '%Y-%m-%d').date()
                T_est_yrs = max(0, (exp_d_dt - today).days) / 365.0

                bsm_est = self.black_scholes_merton(S0, strike_p_num, T_est_yrs, r_est, q_est, vol_est, option_type_p)

                if np.isnan(bsm_est) or bsm_est <= 0:
                    raise ValueError(f"Could not estimate a valid price for {option_type_p.capitalize()} K={strike_p_num} using BSM.")
                market_price_num = bsm_est
                print(f"BSM Estimated Price: {self._format_currency(market_price_num, currency)}")

            return market_price_num

        # --- Get Strategy Specific Parameters ---
        strategy_legs = []
        strategy_name = strategies[strategy_choice]
        max_profit, max_loss = np.nan, np.nan
        breakevens_list = [] # Use list

        try:
            # --- Covered Call [1] ---
            if strategy_choice == 1:
                print(f"\n--- {strategy_name} Setup ---"); print(f"Action: Buy Stock + Sell Call.")
                K_call = safe_float(input(f"Enter Call Strike (e.g., >{S0:.2f}): ").strip())
                if np.isnan(K_call) or K_call <= 0: raise ValueError("Invalid strike.")
                call_premium = get_numeric_option_price(K_call, 'call', expiration_date)
                strategy_legs = [{'type': 'stock', 'dir': 'long', 'price': S0},
                                 {'type': 'call', 'dir': 'short', 'K': K_call, 'price': call_premium}]
                cost_basis = S0 - call_premium
                breakevens_list.append(cost_basis)
                max_profit = K_call - cost_basis
                max_loss = -cost_basis
                print(f"Net Cost Basis (Stock Price - Premium): {self._format_currency(cost_basis, currency)}")

            # --- Protective Put [2] ---
            elif strategy_choice == 2:
                print(f"\n--- {strategy_name} Setup ---"); print(f"Action: Buy Stock + Buy Put.")
                K_put = safe_float(input(f"Enter Put Strike (e.g., <{S0:.2f}): ").strip())
                if np.isnan(K_put) or K_put <= 0: raise ValueError("Invalid strike.")
                put_premium = get_numeric_option_price(K_put, 'put', expiration_date)
                strategy_legs = [{'type': 'stock', 'dir': 'long', 'price': S0},
                                 {'type': 'put', 'dir': 'long', 'K': K_put, 'price': put_premium}]
                total_cost = S0 + put_premium
                breakevens_list.append(total_cost)
                max_profit = np.inf
                max_loss = -(total_cost - K_put)
                print(f"Total Cost (Stock Price + Premium): {self._format_currency(total_cost, currency)}")

            # --- Bull Call Spread [3] ---
            elif strategy_choice == 3:
                print(f"\n--- {strategy_name} Setup ---"); print(f"Action: Buy Lower Call + Sell Higher Call.")
                K_low = safe_float(input(f"Enter Lower Call Strike (Long, e.g., ~{S0*0.98:.2f}): ").strip())
                K_high = safe_float(input(f"Enter Higher Call Strike (Short, e.g., ~{S0*1.02:.2f}): ").strip())
                if not (0 < K_low < K_high): raise ValueError("Requires 0 < K_low < K_high.")
                prem_low = get_numeric_option_price(K_low, 'call', expiration_date)
                prem_high = get_numeric_option_price(K_high, 'call', expiration_date)
                net_debit = prem_low - prem_high
                strategy_legs = [{'type': 'call', 'dir': 'long', 'K': K_low, 'price': prem_low},
                                 {'type': 'call', 'dir': 'short', 'K': K_high, 'price': prem_high}]
                max_profit = (K_high - K_low) - net_debit
                max_loss = -net_debit
                breakevens_list.append(K_low + net_debit)
                if net_debit > 0: print(f"Net Debit: {self._format_currency(net_debit, currency)}")
                else: print(f"Net Credit: {self._format_currency(abs(net_debit), currency)}")

            # --- Bear Put Spread [4] ---
            elif strategy_choice == 4:
                print(f"\n--- {strategy_name} Setup ---"); print(f"Action: Buy Higher Put + Sell Lower Put.")
                K_high = safe_float(input(f"Enter Higher Put Strike (Long, e.g., ~{S0*1.02:.2f}): ").strip())
                K_low = safe_float(input(f"Enter Lower Put Strike (Short, e.g., ~{S0*0.98:.2f}): ").strip())
                if not (0 < K_low < K_high): raise ValueError("Requires 0 < K_low < K_high.")
                prem_high = get_numeric_option_price(K_high, 'put', expiration_date)
                prem_low = get_numeric_option_price(K_low, 'put', expiration_date)
                net_debit = prem_high - prem_low
                strategy_legs = [{'type': 'put', 'dir': 'long', 'K': K_high, 'price': prem_high},
                                 {'type': 'put', 'dir': 'short', 'K': K_low, 'price': prem_low}]
                max_profit = (K_high - K_low) - net_debit
                max_loss = -net_debit
                breakevens_list.append(K_high - net_debit)
                if net_debit > 0: print(f"Net Debit: {self._format_currency(net_debit, currency)}")
                else: print(f"Net Credit: {self._format_currency(abs(net_debit), currency)}")

            # --- Long Straddle [5] ---
            elif strategy_choice == 5:
                print(f"\n--- {strategy_name} Setup ---"); print(f"Action: Buy ATM Call + Buy ATM Put.")
                # Find ATM strike robustly
                try:
                    options = stock_data['ticker_object'].option_chain(expiration_date)
                    strikes_series = pd.concat([options.calls['strike'], options.puts['strike']]).dropna().unique()
                    strikes_num = sorted([s for s in strikes_series if pd.notna(safe_float(s))])
                    if not strikes_num: raise ValueError("No valid numeric strikes found.")
                    K_atm = min(strikes_num, key=lambda x: abs(x - S0))
                    print(f"Using closest ATM strike: {self._format_currency(K_atm, currency)}")
                except Exception as e: raise ValueError(f"Could not determine ATM strike: {e}.")
                prem_call = get_numeric_option_price(K_atm, 'call', expiration_date)
                prem_put = get_numeric_option_price(K_atm, 'put', expiration_date)
                total_cost = prem_call + prem_put
                strategy_legs = [{'type': 'call', 'dir': 'long', 'K': K_atm, 'price': prem_call},
                                 {'type': 'put', 'dir': 'long', 'K': K_atm, 'price': prem_put}]
                max_profit = np.inf
                max_loss = -total_cost
                breakevens_list.extend([K_atm - total_cost, K_atm + total_cost])
                print(f"Total Cost (Net Debit): {self._format_currency(total_cost, currency)}")

            # --- Long Strangle [6] ---
            elif strategy_choice == 6:
                print(f"\n--- {strategy_name} Setup ---"); print(f"Action: Buy OTM Call + Buy OTM Put.")
                K_call = safe_float(input(f"Enter OTM Call Strike (Long, e.g., >{S0*1.05:.2f}): ").strip())
                K_put = safe_float(input(f"Enter OTM Put Strike (Long, e.g., <{S0*0.95:.2f}): ").strip())
                if not (0 < K_put < K_call): print("Warning: Usually K_put < K_call for strangle.")
                if np.isnan(K_call) or np.isnan(K_put) or K_call <=0 or K_put <=0: raise ValueError("Invalid strike(s).")
                prem_call = get_numeric_option_price(K_call, 'call', expiration_date)
                prem_put = get_numeric_option_price(K_put, 'put', expiration_date)
                total_cost = prem_call + prem_put
                strategy_legs = [{'type': 'call', 'dir': 'long', 'K': K_call, 'price': prem_call},
                                 {'type': 'put', 'dir': 'long', 'K': K_put, 'price': prem_put}]
                max_profit = np.inf
                max_loss = -total_cost
                breakevens_list.extend([K_put - total_cost, K_call + total_cost])
                print(f"Total Cost (Net Debit): {self._format_currency(total_cost, currency)}")

            # --- Common Calculation & Plotting ---
            if strategy_legs:
                # Define price range, ensuring it covers critical points
                crit_points = [S0] + [leg['K'] for leg in strategy_legs if pd.notna(safe_float(leg.get('K')))] + breakevens_list
                valid_points = [p for p in crit_points if pd.notna(safe_float(p))]
                if not valid_points: valid_points = [S0] # Fallback if no strikes/BEs

                price_range_factor = self.config['strategy_price_range']
                S_T_min = max(0, min(valid_points) * (1 - price_range_factor * 1.5)) # Extend range further
                S_T_max = max(valid_points) * (1 + price_range_factor * 1.5)
                S_T_range = np.linspace(S_T_min, S_T_max, 150)

                PnL = np.array([self._calculate_payoff(s_t, strategy_legs) for s_t in S_T_range])

                # --- Display Summary ---
                print("\n--- Strategy Summary ---")
                print(f"Strategy: {strategy_name} | Expiration: {expiration_date}")
                print(f"Current Underlying Price: {self._format_currency(S0, currency)}")
                print("Legs:")
                net_cost_summary = 0
                for i, leg in enumerate(strategy_legs):
                     k_str = f" K={self._format_currency(leg['K'], currency)}" if pd.notna(safe_float(leg.get('K'))) else ""
                     p_str = f" @ {self._format_currency(leg['price'], currency)}"
                     print(f"  {i+1}: {leg['dir'].capitalize()} {leg['type'].capitalize()}{k_str}{p_str}")
                     if leg['dir']=='long': net_cost_summary += safe_float(leg['price'],0)
                     else: net_cost_summary -= safe_float(leg['price'],0)

                if net_cost_summary > 1e-6: print(f"Net Cost (Debit): {self._format_currency(net_cost_summary, currency)}")
                elif net_cost_summary < -1e-6: print(f"Net Credit: {self._format_currency(abs(net_cost_summary), currency)}")
                else: print(f"Net Cost: ~Zero")

                be_str = ", ".join([self._format_currency(be, currency) for be in sorted(breakevens_list)]) or "N/A"
                mp_str = self._format_currency(max_profit, currency) if np.isfinite(max_profit) else ('Unlimited' if max_profit > 0 else 'N/A')
                ml_str = self._format_currency(max_loss, currency) if np.isfinite(max_loss) else ('Unlimited' if max_loss < 0 else 'N/A')

                print(f"\nBreakeven(s) at Expiration: {be_str}")
                print(f"Maximum Potential Profit: {mp_str}")
                print(f"Maximum Potential Loss: {ml_str}")

                # --- Plot Payoff Diagram ---
                print("\nGenerating Payoff Diagram...")
                self._plot_payoff(S_T_range, PnL, strategy_name, breakevens_list, max_profit, max_loss, currency)

        except ValueError as ve: print(f"\nInput/Calculation Error: {ve}")
        except Exception as e:
             print(f"\nAnalysis Error: {e}")
             if self.config['debug_mode']: traceback.print_exc()

    # --- Menu and Application Flow (Console) ---

    def manage_favorites(self):
        """Manage the list of favorite tickers (Console)."""
        while True:
            self.clear_screen(); print("--- Manage Favorite Tickers ---")
            if not self.favorite_tickers: print("No favorites saved.")
            else:
                print("Current Favorites:"); [print(f" {i+1}. {t}") for i, t in enumerate(self.favorite_tickers)]
            print("\nOptions:\n 1. Add Ticker\n 2. Remove Ticker\n 0. Back")
            choice = input("Enter option: ").strip()
            if choice == '1':
                ticker_add = input("Enter ticker to add: ").strip().upper()
                if ticker_add and self.validate_ticker(ticker_add):
                    if ticker_add not in self.favorite_tickers:
                        self.favorite_tickers.append(ticker_add); self.favorite_tickers.sort()
                        self._save_favorite_tickers(); print(f"'{ticker_add}' added.")
                    else: print(f"'{ticker_add}' already in favorites.")
                elif ticker_add: print(f"Could not validate '{ticker_add}'. Not added.")
                input("Press Enter...");
            elif choice == '2':
                if not self.favorite_tickers: print("No favorites to remove."); input("Press Enter..."); continue
                try:
                    num_rem = int(input("Enter number to remove: ").strip())
                    if 1 <= num_rem <= len(self.favorite_tickers):
                        removed = self.favorite_tickers.pop(num_rem - 1)
                        self._save_favorite_tickers(); print(f"'{removed}' removed.")
                    else: print("Invalid number.")
                except ValueError: print("Invalid input.")
                input("Press Enter...");
            elif choice == '0': break
            else: print("Invalid option."); input("Press Enter...");

    def manage_settings(self):
         """Allow user to view and modify config settings (Console)."""
         while True:
             self.clear_screen(); print("--- Configure Settings ---")
             settings_list = list(self.config.items())
             for i, (k, v) in enumerate(settings_list): print(f" {i+1}. {k}: {v}")
             print("\n 0. Back to Main Menu (Save Changes)")
             choice_str = input("\nEnter number to change (or 0 to save/exit): ").strip()
             try:
                  choice_idx = int(choice_str)
                  if choice_idx == 0: self._save_config(); break
                  elif 1 <= choice_idx <= len(settings_list):
                       key, current_val = settings_list[choice_idx - 1]
                       if isinstance(current_val, (dict, list)):
                           print("Editing complex settings not supported here."); input("Press Enter..."); continue
                       new_val_str = input(f"Enter new value for '{key}' (current: {current_val}): ").strip()
                       try: # Convert to the correct type
                            target_type = type(current_val)
                            if target_type == bool:
                                 if new_val_str.lower() in ['true','t','yes','y','1']: new_val = True
                                 elif new_val_str.lower() in ['false','f','no','n','0']: new_val = False
                                 else: raise ValueError("Use true/false for boolean")
                            else: new_val = target_type(new_val_str) # Handles int, float, str

                            # --- Add Validation Rules Here ---
                            valid = True
                            if key == 'max_strikes_chain' and not (5 <= new_val <= 100): valid = False; print("Max strikes must be 5-100.")
                            elif key == 'default_risk_free_rate' and not (0 <= new_val <= 0.5): valid = False; print("Rate must be 0%-50%.")
                            # Add more rules...

                            if valid: self.config[key] = new_val; print("Setting updated.")
                            else: print("Invalid value. Setting not changed.")
                       except ValueError: print(f"Invalid type. Expected {target_type.__name__}.")
                       input("Press Enter...");
                  else: print("Invalid selection."); input("Press Enter...");
             except ValueError: print("Invalid input."); input("Press Enter...");

    def display_main_menu(self):
        """Display the main console menu options."""
        self.clear_screen()
        print("+" + "=" * 35 + "+")
        print("|     Options Analyzer Menu         |")
        print("+" + "=" * 35 + "+")
        curr_price_disp = "N/A"
        if self.current_ticker and self.current_stock_data:
             curr_price_disp = self._format_currency(self.current_stock_data['current_price'], self.current_stock_data.get('currency','USD'))
             print(f" Current Ticker: {self.current_ticker} ({curr_price_disp})")
        else: print(" Current Ticker: None")
        print("-" * 37)
        print("  1. Fetch Stock Data / Change Ticker")
        print("  2. Simple Option Price (BSM & Greeks)")
        print("  3. View Options Chain (Table & Graph)")
        print("  4. Analyze Option Strategy (Payoff)")
        print("  5. Manage Favorite Tickers")
        print("  6. Configure Settings")
        print("  0. Exit")
        print("-" * 37)
        if self.favorite_tickers:
            if len(self.favorite_tickers) > 5:
                fav_str += "..." # Changed from invalid list comprehension syntax
            print(f" Favs: {fav_str}")
        print("+" + "=" * 35 + "+")

    def run(self):
        """Main console application loop."""
        print("Welcome to the Enhanced Options Analyzer (Corrected)!")
        while True:
            self.display_main_menu()
            choice = input("Enter your choice: ").strip()
            try:
                if choice == '1':
                    prompt = "Enter ticker"; opts = []; curr = self.current_ticker
                    if curr: opts.append(f"Enter='{curr}'")
                    if self.favorite_tickers: opts.append("'fav'=list")
                    if opts: prompt += f" ({', '.join(opts)})"
                    ticker_in = input(f"{prompt}: ").strip().upper()
                    sel_ticker = None
                    if not ticker_in and curr: sel_ticker = curr; print(f"Refreshing {curr}..."); self.get_stock_data(sel_ticker)
                    elif ticker_in == 'FAV' and self.favorite_tickers:
                         print("Favs:"); [print(f" {i+1}. {t}") for i,t in enumerate(self.favorite_tickers)]
                         fav_in = input("Select number: ").strip()
                         idx = int(fav_in) - 1
                         if 0 <= idx < len(self.favorite_tickers): sel_ticker = self.favorite_tickers[idx]
                         else: print("Invalid selection.")
                    elif ticker_in: sel_ticker = ticker_in
                    if sel_ticker and sel_ticker != self.current_ticker: self.get_stock_data(sel_ticker)
                    elif not sel_ticker and not ticker_in and not curr: print("No ticker entered.")
                elif choice == '2': self.get_simple_option_price()
                elif choice == '3': self.calculate_options_chain() # Let user choose viz inside
                elif choice == '4': self.analyze_strategy()
                elif choice == '5': self.manage_favorites()
                elif choice == '6': self.manage_settings()
                elif choice == '0': print("\nExiting Options Analyzer. Goodbye!"); break
                else: print("Invalid choice. Please try again.")
            except Exception as e: # Catch unexpected errors during menu actions
                 print(f"\nAn unexpected error occurred in menu action: {e}")
                 if self.config.get('debug_mode', False): traceback.print_exc()

            if choice != '0': input("\nPress Enter to return to the Main Menu...")

if __name__ == "__main__":
    analyzer = OptionsAnalyzer()
    analyzer.run()