# options_analyzer_combined.py
# Enhanced Options Pricing and Analysis Tool (Console Version)

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

# Set plotting style
style.use('seaborn-v0_8-darkgrid') # Using a potentially 'prettier' style

class OptionsAnalyzer:
    """
    A tool for fetching stock data, calculating option prices (Black-Scholes-Merton
    with dividends), analyzing option chains, calculating Greeks, implied volatility,
    and evaluating common option strategies with payoff diagrams via a console interface.
    """
    def __init__(self):
        """Initialize the Options Analyzer with default parameters"""
        self.current_ticker = None
        self.current_stock_data = None
        self.risk_free_rate = None
        self.config = self._load_config()
        self.favorite_tickers = self._load_favorite_tickers()
        # Fetch initial risk-free rate (can be silent or verbose based on need)
        self.get_risk_free_rate(verbose=True) # Set verbose=True for console

    # --- Configuration and Persistence ---

    def _load_config(self):
        """Load configuration from file or use defaults."""
        default_config = {
            'volatility_days': 252,  # Trading days for annualization
            'default_risk_free_rate': 0.04,  # 4% fallback
            'show_greeks_in_chain': True,
            'max_strikes_chain': 20,  # Max strikes around ATM in chain display (increased slightly)
            'iv_precision': 0.0001,  # Precision for implied volatility calculation
            'iv_max_iterations': 100, # Max iterations for IV calculation
            'strategy_price_range': 0.3, # +/- 30% range for payoff diagrams
            'debug_mode': False
        }
        config_path = 'options_config.json'
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Add any missing keys from default config
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    print("Configuration loaded.")
                    return config
            else:
                print("No config file found, using defaults.")
                return default_config
        except json.JSONDecodeError:
            print(f"Error reading config file '{config_path}'. Using defaults.")
            return default_config
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return default_config

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
                    print(f"Loaded {len(favorites)} favorite tickers.")
                    return favorites
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
        if pd.isna(value):
            return "N/A"
        try:
            # Basic currency symbols - could be expanded
            symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
            symbol = symbols.get(currency, '')
            return f"{symbol}{value:,.2f}"
        except (TypeError, ValueError):
             return "N/A" # Handle non-numeric inputs gracefully

    def validate_ticker(self, ticker):
        """Validate if the ticker exists using yfinance."""
        if not ticker:
            print("Ticker symbol cannot be empty.")
            return False
        try:
            print(f"\nValidating ticker '{ticker}'...")
            stock = yf.Ticker(ticker)
            info = stock.info
            # Check if info is substantially empty or market state indicates issues
            if not info or info.get('quoteType') == 'MUTUALFUND' or info.get('marketState') == 'POSTPOST':
                 # Fallback check using history
                 hist = stock.history(period="5d")
                 if hist.empty:
                     print(f"Ticker '{ticker}' is not valid or no recent data available.")
                     return False
            # Basic check if it's likely an equity or ETF
            q_type = info.get('quoteType', 'N/A')
            if q_type not in ['EQUITY', 'ETF']:
                 print(f"Warning: Ticker '{ticker}' may not be an equity or ETF ({q_type}). Options might not be available.")
                 # Allow proceeding but warn the user

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
                exp_date = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
                days = (exp_date - today).days
                if days >= 0: # Only show non-expired dates
                    print(f"{len(valid_expirations) + 1}. {date_str} ({days} days)")
                    valid_expirations.append({'index': i, 'date': date_str, 'days': days})
            except ValueError:
                 if self.config['debug_mode']:
                     print(f"Skipping invalid date format: {date_str}")
                 continue # Skip invalid date formats

        if not valid_expirations:
            print("No valid future expiration dates found.")
            return None

        while True:
            try:
                selection = input(f"\nSelect expiration date (1-{len(valid_expirations)}), Enter for first: ")
                if not selection: # Handle empty input -> default to first
                    selected_exp = valid_expirations[0]
                    print(f"Using first available date: {selected_exp['date']}")
                    break
                idx = int(selection) - 1
                if 0 <= idx < len(valid_expirations):
                    selected_exp = valid_expirations[idx]
                    break
                else:
                    print("Invalid selection. Please enter a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"\nSelected expiration date: {selected_exp['date']} ({selected_exp['days']} days)")
        return selected_exp['date']

    # --- Data Fetching ---

    def get_stock_data(self, ticker):
        """Fetch stock data, company info, dividend yield, and options expirations."""
        ticker = ticker.upper().strip()
        if not self.validate_ticker(ticker):
            return None

        try:
            print(f"\nFetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y") # For volatility

            current_price = None
            volatility = None

            # Calculate Volatility from History
            if not hist.empty:
                current_price = hist['Close'].iloc[-1] # Use last close from history
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                if len(returns) >= 2:
                    volatility = returns.std() * np.sqrt(self.config['volatility_days'])
                else:
                    print(f"Warning: Not enough history points for {ticker} volatility calc.")
            else:
                print(f"Warning: Could not fetch sufficient historical data for {ticker}.")

            # Get Company Info, Fallback Price, and Dividend Yield
            info = {}
            company_name = ticker
            sector = 'N/A'
            industry = 'N/A'
            market_cap_str = 'N/A'
            currency = 'USD'
            dividend_yield = 0.0 # Default

            try:
                info = stock.info
                company_name = info.get('shortName', ticker)
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                market_cap = info.get('marketCap')
                currency = info.get('currency', 'USD')
                # Fetch dividend yield, default 0 if None or not present
                dividend_yield = info.get('dividendYield') or 0.0

                # Fallback for current price if history failed
                if current_price is None:
                     current_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
                     if current_price:
                         print("Using current/previous close price from quote info.")
                     else:
                         # If still no price, try day's range (less ideal)
                         high = info.get('dayHigh')
                         low = info.get('dayLow')
                         if high and low:
                             current_price = (high + low) / 2
                             print("Warning: Using midpoint of day's high/low as current price.")

                # Format Market Cap
                if market_cap:
                    if market_cap >= 1e12: market_cap_str = f"{self._format_currency(market_cap / 1e12, currency)}T"
                    elif market_cap >= 1e9: market_cap_str = f"{self._format_currency(market_cap / 1e9, currency)}B"
                    elif market_cap >= 1e6: market_cap_str = f"{self._format_currency(market_cap / 1e6, currency)}M"
                    else: market_cap_str = self._format_currency(market_cap, currency)

            except Exception as e_info:
                print(f"Warning: Could not fetch some company info details: {e_info}")
                if self.config['debug_mode']: traceback.print_exc()

            # Check if price was determined
            if current_price is None:
                 raise ValueError(f"CRITICAL: Failed to determine current price for {ticker} from history or info.")

            # Print stock details
            print(f"\n=== {company_name} ({ticker}) ===")
            print(f"Current price: {self._format_currency(current_price, currency)}")
            print(f"Sector: {sector} | Industry: {industry}")
            print(f"Market Cap: {market_cap_str}")
            print(f"Dividend Yield: {dividend_yield:.4f} ({dividend_yield*100:.2f}%)") # Display yield
            if volatility is not None:
                print(f"Annualized Volatility (1y): {volatility:.4f} ({volatility*100:.2f}%)")
            else:
                 print("Annualized Volatility (1y): N/A (Insufficient history)")

            # Get available expiration dates
            expirations = ()
            try:
                 expirations = stock.options
                 if not expirations:
                      print("Note: No options expiration dates found for this ticker.")
            except Exception as e:
                 print(f"Warning: Could not fetch options expiration dates: {e}")


            self.current_ticker = ticker
            self.current_stock_data = {
                'ticker': ticker,
                'current_price': current_price,
                'volatility': volatility, # Can be None
                'dividend_yield': float(dividend_yield) if dividend_yield is not None else 0.0, # Store yield
                'expirations': expirations,
                'ticker_object': stock,
                'history': hist, # Can be empty
                'info': info, # Store info dict
                'currency': currency
            }

            print(f"\nData fetch complete for {ticker}.")
            return self.current_stock_data

        except Exception as e:
            print(f"\nError fetching data for '{ticker}': {e}")
            if self.config['debug_mode']:
                traceback.print_exc()
            # Reset current data if fetch fails
            if self.current_ticker == ticker:
                 self.current_ticker = None
                 self.current_stock_data = None
            return None

    def get_risk_free_rate(self, verbose=False):
        """Get risk-free rate from Treasury yield (^TNX) or default."""
        try:
            if verbose: print("Fetching current risk-free rate (10-Year Treasury Yield)...")
            treasury = yf.Ticker("^TNX")
            data = treasury.history(period="5d") # Fetch a few days for robustness
            if not data.empty:
                rate = data['Close'].iloc[-1] / 100 # Convert percentage to decimal
                # Basic sanity check for the rate
                if 0 <= rate <= 0.2: # Assume rate won't be negative or > 20%
                    if verbose: print(f"Using current risk-free rate (10Y Treasury): {rate:.4f} ({rate*100:.2f}%)")
                    self.risk_free_rate = rate
                    return rate
                else:
                     if verbose: print(f"Warning: Fetched treasury rate ({rate:.4f}) seems unusual. Falling back to default.")
            else:
                 if verbose: print("Could not fetch treasury data. Falling back to default.")

        except Exception as e:
            if verbose: print(f"Error fetching risk-free rate: {e}. Falling back to default.")
            if self.config['debug_mode']:
                traceback.print_exc()

        # Fallback to default
        default_rate = self.config['default_risk_free_rate']
        if verbose: print(f"Using default risk-free rate: {default_rate:.4f} ({default_rate*100:.2f}%)")
        self.risk_free_rate = default_rate
        return default_rate

    def _get_option_data_for_strike(self, expiration_date, strike, option_type):
         """Helper to get specific option data (call or put) for a strike."""
         if not self.current_stock_data or not self.current_stock_data.get('ticker_object'):
             print("Error: Stock data not loaded.")
             return None
         if not expiration_date:
              print("Error: Expiration date required.")
              return None

         stock = self.current_stock_data['ticker_object']
         option_type = option_type.lower()

         try:
             # Fetch or re-use option chain for the date
             # Simple caching idea (could be more robust)
             cache_key = f"{expiration_date}_{option_type}"
             if not hasattr(self, '_chain_cache') or self._chain_cache.get('date') != expiration_date:
                 print(f"Fetching chain for {expiration_date}...")
                 opt_chain = stock.option_chain(expiration_date)
                 self._chain_cache = {'date': expiration_date, 'calls': opt_chain.calls, 'puts': opt_chain.puts}
             elif cache_key not in self._chain_cache:
                 # If date matches but type wasn't fetched yet (unlikely with outer merge but safe)
                 print(f"Fetching chain for {expiration_date}...")
                 opt_chain = stock.option_chain(expiration_date)
                 self._chain_cache = {'date': expiration_date, 'calls': opt_chain.calls, 'puts': opt_chain.puts}


             data = self._chain_cache['calls'] if option_type == 'call' else self._chain_cache['puts']

             # Find the specific strike row
             option_data = data[data['strike'] == strike]

             if option_data.empty:
                 #print(f"Warning: No {option_type} data found for strike {strike} on {expiration_date}.") # Reduced verbosity
                 return None

             # Return the first row (as a Series)
             return option_data.iloc[0]

         except IndexError: # Handle cases where yfinance might return unexpected empty structures
              print(f"Error: No options data structure available for {self.current_ticker} on {expiration_date}.")
              self._chain_cache = {} # Clear cache on error
              return None
         except Exception as e:
             print(f"Error fetching option data for strike {strike} ({option_type}): {e}")
             if self.config['debug_mode']:
                 traceback.print_exc()
             self._chain_cache = {} # Clear cache on error
             return None

    # --- Black-Scholes-Merton Model and Greeks (with Dividend Yield q) ---

    def black_scholes_merton(self, S, K, T, r, q, sigma, option_type="call"):
        """ BSM Option Price including continuous dividend yield q. """
        if T < 0: T = 0 # Handle expiration
        if sigma <= 0: sigma = 1e-6 # Avoid division by zero, use minimal vol
        if S <=0 or K <= 0: return np.nan # Prices must be positive

        # Handle immediate expiration (intrinsic value)
        if T == 0:
            if option_type.lower() == "call": return max(0.0, S - K)
            elif option_type.lower() == "put": return max(0.0, K - S)
            else: return np.nan

        # Calculate d1 and d2 using dividend yield q
        try:
            # Check for potential issues in log or sqrt
            if S <= 0 or K <= 0: return np.nan
            sqrt_T = np.sqrt(T)
            if sigma * sqrt_T == 0: return np.nan # Avoid division by zero

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

            if option_type.lower() == "call":
                # C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == "put":
                # P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            else:
                print(f"Warning: Invalid option type '{option_type}' in BSM.")
                return np.nan
            return max(0.0, price) # Price cannot be negative

        except (OverflowError, ValueError) as e:
            # Catch math errors like log(negative) or overflow
            # print(f"Warning: Math error in BSM calculation: {e}") # Debug only
            return np.nan
        except Exception as e:
            print(f"Error in BSM calculation: {e}")
            if self.config['debug_mode']: traceback.print_exc()
            return np.nan

    def calculate_option_greeks(self, S, K, T, r, q, sigma, option_type="call"):
        """ Calculate option Greeks including continuous dividend yield q. """
        greeks = { "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan }
        # Input validation
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            # Handle greeks at expiration (some might be defined, but BSM less applicable)
            if T <= 0:
                if option_type.lower() == 'call':
                    greeks['delta'] = 1.0 if S > K else (0.5 if S == K else 0.0)
                elif option_type.lower() == 'put':
                    greeks['delta'] = -1.0 if S < K else (-0.5 if S == K else 0.0)
                greeks['gamma'] = 0.0
                greeks['theta'] = 0.0
                greeks['vega'] = 0.0
                greeks['rho'] = 0.0
            return greeks # Return NaNs or zeros for other invalid inputs

        option_type = option_type.lower()
        try:
            sqrt_T = np.sqrt(T)
            exp_qT = np.exp(-q * T)
            exp_rT = np.exp(-r * T)

            # Check for potential division by zero before d1/d2
            if sigma * sqrt_T == 0: return {k: 0.0 for k in greeks} # Greeks are zero if no vol or time

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_neg_d2 = norm.cdf(-d2)

            # Gamma: N'(d1) * exp(-qT) / (S * sigma * sqrt(T))
            greeks["gamma"] = exp_qT * pdf_d1 / (S * sigma * sqrt_T) if (S * sigma * sqrt_T) != 0 else 0

            # Vega: S * N'(d1) * sqrt(T) * exp(-qT) / 100 (per 1% change)
            greeks["vega"] = (S * exp_qT * sqrt_T * pdf_d1) / 100

            # Theta (per day): Complex formula, includes terms for time decay, interest, and dividend
            # Theta = -[S*N'(d1)*sigma*exp(-qT)] / [2*sqrt(T)] - r*K*exp(-rT)*N(d2) + q*S*exp(-qT)*N(d1) (for Call)
            # Theta = -[S*N'(d1)*sigma*exp(-qT)] / [2*sqrt(T)] + r*K*exp(-rT)*N(-d2) - q*S*exp(-qT)*N(-d1) (for Put)
            theta_term1 = - (S * exp_qT * pdf_d1 * sigma) / (2 * sqrt_T)

            if option_type == "call":
                # Delta: exp(-qT) * N(d1)
                greeks["delta"] = exp_qT * cdf_d1
                # Theta
                theta_term2 = - r * K * exp_rT * cdf_d2
                theta_term3 = + q * S * exp_qT * cdf_d1
                greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
                # Rho: K * T * exp(-rT) * N(d2) / 100 (per 1% change in r)
                greeks["rho"] = (K * T * exp_rT * cdf_d2) / 100
            elif option_type == "put":
                # Delta: exp(-qT) * (N(d1) - 1)
                greeks["delta"] = exp_qT * (cdf_d1 - 1)
                # Theta
                theta_term2 = + r * K * exp_rT * cdf_neg_d2 # Note the '+' and -d2
                theta_term3 = - q * S * exp_qT * cdf_neg_d1 # Note the '-' and -d1
                greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
                # Rho: -K * T * exp(-rT) * N(-d2) / 100 (per 1% change in r)
                greeks["rho"] = (-K * T * exp_rT * cdf_neg_d2) / 100
            else:
                print(f"Warning: Invalid option type '{option_type}' for Greeks.")
                return {k: np.nan for k in greeks}

            return greeks

        except (ZeroDivisionError, OverflowError, ValueError) as e:
            # print(f"Warning: Math error in Greeks calculation: {e}") # Debug only
            # Return NaNs as calculation failed
            return {k: np.nan for k in greeks}
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            if self.config['debug_mode']: traceback.print_exc()
            return {k: np.nan for k in greeks}

    def calculate_implied_volatility(self, S, K, T, r, q, market_price, option_type="call"):
        """ Calculate implied volatility using bisection (includes dividend yield q). """
        option_type = option_type.lower()
        precision = self.config['iv_precision']
        max_iterations = self.config['iv_max_iterations']

        # --- Input Validation and Edge Cases ---
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return np.nan # Cannot calculate IV

        # Check if market price is below discounted intrinsic value
        intrinsic_value = 0.0
        try:
            if option_type == "call":
                intrinsic_value = max(0.0, S * np.exp(-q*T) - K * np.exp(-r * T))
            elif option_type == "put":
                intrinsic_value = max(0.0, K * np.exp(-r * T) - S * np.exp(-q*T))
            else:
                 print(f"Warning: Invalid option type '{option_type}' for IV.")
                 return np.nan
        except OverflowError:
            return np.nan # Cannot calculate intrinsic if inputs cause overflow

        # Allow a small tolerance for rounding errors in market data
        if market_price < intrinsic_value - precision:
            # print(f"Debug: Market price ({market_price:.4f}) below intrinsic ({intrinsic_value:.4f}) for {option_type} K={K}. Returning NaN.") # Debug only
            return np.nan

        # --- Bisection Method ---
        vol_low = 1e-5  # Lower bound (near zero)
        vol_high = 5.0  # Upper bound (500% vol)

        # Calculate price at bounds using the BSM function with dividend yield
        price_low = self.black_scholes_merton(S, K, T, r, q, vol_low, option_type)
        price_high = self.black_scholes_merton(S, K, T, r, q, vol_high, option_type)

        # Check if BSM calculation failed at bounds
        if pd.isna(price_low) or pd.isna(price_high):
             # print(f"Warning: BSM returned NaN at IV bounds for K={K}. Cannot calculate IV.") # Debug only
             return np.nan

        # Check if market price is outside the possible range given vol bounds
        if market_price <= price_low: return vol_low # Closest is minimum vol
        if market_price >= price_high: return vol_high # Closest is maximum vol

        # Bisection search loop
        vol_mid = vol_low # Initialize vol_mid
        for _ in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            if vol_mid <= 0: vol_mid = 1e-5 # Ensure vol doesn't become zero/negative

            price_mid = self.black_scholes_merton(S, K, T, r, q, vol_mid, option_type)

            if pd.isna(price_mid):
                 # BSM failed at mid-volatility, often due to extreme inputs or instability.
                 # Hard to reliably recover, safest to return NaN.
                 # print(f"Warning: BSM returned NaN at vol_mid={vol_mid} for K={K}. Cannot converge IV.") # Debug only
                 return np.nan

            # Check for convergence
            if abs(price_mid - market_price) < precision:
                return vol_mid

            # Narrow the search interval
            if price_mid > market_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid

            # Check if bounds are too close (avoid infinite loops)
            if abs(vol_high - vol_low) < precision / 10: # Use tighter precision for bound check
                 break

        # If max_iterations reached, return the midpoint as best estimate
        # But perform a final check if the price is reasonably close
        final_vol = (vol_low + vol_high) / 2
        final_price = self.black_scholes_merton(S, K, T, r, q, final_vol, option_type)
        if pd.notna(final_price) and abs(final_price - market_price) < precision * 10: # Looser check
             return final_vol
        else:
             # print(f"Warning: IV calculation did not converge sufficiently for K={K}. Market: {market_price:.2f}, Model: {final_price:.2f} at IV {final_vol*100:.2f}%") # Debug only
             return np.nan # Indicate failure to converge reliably

    # --- Core Functionality Methods (Console Focused) ---

    def get_simple_option_price(self):
        """Calculate and display a simple option price based on user input (Console)."""
        if self.current_stock_data is None:
            print("\nPlease fetch stock data first (Option 1).")
            ticker = input("Enter stock ticker symbol: ").upper()
            if not ticker or not self.get_stock_data(ticker):
                 return # Exit if fetching fails or no ticker entered
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change_ticker = input("Fetch data for a different ticker? (y/n, default n): ").lower()
             if change_ticker == 'y':
                  ticker = input("Enter new stock ticker symbol: ").upper()
                  if not ticker or not self.get_stock_data(ticker):
                      return

        # Use currently loaded data
        stock_data = self.current_stock_data
        if not stock_data:
             print("Error: Stock data is not available.")
             return

        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        dividend_yield = stock_data['dividend_yield'] # Get yield
        expirations = stock_data['expirations']
        stock = stock_data['ticker_object']
        currency = stock_data['currency']
        risk_free_rate = self.risk_free_rate # Use fetched rate

        # Handle missing volatility
        if volatility is None:
             print("\nWarning: Historical volatility is not available.")
             try:
                  user_vol = float(input("Enter an estimated annual volatility (e.g., 0.3 for 30%) or press Enter to use 0.3: ") or 0.3)
                  volatility = user_vol if user_vol > 0 else 1e-6 # Use tiny vol if 0 or negative
             except ValueError:
                  print("Invalid input. Using 0.3 volatility.")
                  volatility = 0.3 # Use default estimate if input error

        if risk_free_rate is None: # Should have been fetched, but check again
             risk_free_rate = self.get_risk_free_rate(verbose=True)

        # --- Select Expiration ---
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date: return

        # Calculate time to expiration
        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days) # Ensure non-negative
        T = days_to_expiration / 365.0 # Use 365 for consistency
        print(f"Time to expiration (T): {T:.4f} years ({days_to_expiration} days)")

        # --- Select Strike Price ---
        strike = None
        while strike is None:
             strike_input = input(f"\nEnter strike price (e.g., {current_price:.2f}) or 'atm' for closest: ").lower().strip()
             if strike_input == 'atm':
                try:
                    # Fetch available strikes for the expiration
                    options = stock.option_chain(expiration_date)
                    # Combine call and put strikes, find unique ones, sort
                    all_strikes = sorted(list(set(options.calls['strike'].tolist() + options.puts['strike'].tolist())))
                    if not all_strikes:
                         print("No strikes found for this expiration. Using current price as strike.")
                         strike = current_price
                    else:
                         # Find the strike closest to the current price
                         strike = min(all_strikes, key=lambda x: abs(x - current_price))
                         print(f"Found closest available strike: {self._format_currency(strike, currency)}")
                except Exception as e:
                    print(f"Could not fetch available strikes. Using current price. Error: {e}")
                    strike = current_price # Fallback
             else:
                try:
                    strike = float(strike_input)
                    if strike <= 0:
                         print("Strike price must be positive.")
                         strike = None # Force re-entry
                except ValueError:
                    print("Invalid input. Please enter a number or 'atm'.")

        # --- Select Option Type ---
        option_type = None
        while option_type not in ['call', 'put', 'both']:
            option_type_input = input("Calculate for 'call', 'put', or 'both'? (default 'both'): ").lower().strip()
            if not option_type_input: option_type = 'both'
            elif option_type_input in ['call', 'put', 'both']: option_type = option_type_input
            else: print("Invalid option type.")

        # --- Calculate and Display ---
        results = {}
        print(f"\n--- BSM Option Analysis ---")
        print(f"Stock: {self.current_ticker} @ {self._format_currency(current_price, currency)}")
        print(f"Strike: {self._format_currency(strike, currency)}")
        print(f"Expiration: {expiration_date} ({days_to_expiration} days, T={T:.4f})")
        print(f"Volatility (Input): {volatility*100:.2f}%")
        print(f"Risk-Free Rate: {risk_free_rate*100:.2f}%")
        print(f"Dividend Yield: {dividend_yield*100:.2f}%") # Show yield used
        print("-" * 30)

        if option_type in ['call', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "call")
            greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "call")
            print(f"BSM Call Price: {self._format_currency(bsm_price, currency)}")
            if greeks:
                print("  Greeks (Call):")
                print(f"    Delta: {greeks['delta']:.4f}")
                print(f"    Gamma: {greeks['gamma']:.4f}")
                print(f"    Theta: {self._format_currency(greeks['theta'], currency)} / day") # Daily Theta
                print(f"    Vega:  {self._format_currency(greeks['vega'], currency)} / 1% vol")
                print(f"    Rho:   {self._format_currency(greeks['rho'], currency)} / 1% rate")
            print("-" * 30)

        if option_type in ['put', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "put")
            greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, volatility, "put")
            print(f"BSM Put Price: {self._format_currency(bsm_price, currency)}")
            if greeks:
                print("  Greeks (Put):")
                print(f"    Delta: {greeks['delta']:.4f}")
                print(f"    Gamma: {greeks['gamma']:.4f}")
                print(f"    Theta: {self._format_currency(greeks['theta'], currency)} / day") # Daily Theta
                print(f"    Vega:  {self._format_currency(greeks['vega'], currency)} / 1% vol")
                print(f"    Rho:   {self._format_currency(greeks['rho'], currency)} / 1% rate")
            print("-" * 30)


    def calculate_options_chain(self, visualize=False):
        """Calculate and display a detailed options chain for a selected expiration (Console)."""
        if self.current_stock_data is None:
             print("\nPlease fetch stock data first (Option 1).")
             ticker = input("Enter stock ticker symbol: ").upper()
             if not ticker or not self.get_stock_data(ticker): return None
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change_ticker = input("Fetch data for a different ticker? (y/n, default n): ").lower()
             if change_ticker == 'y':
                  ticker = input("Enter new stock ticker symbol: ").upper()
                  if not ticker or not self.get_stock_data(ticker): return None

        # Use currently loaded data
        stock_data = self.current_stock_data
        if not stock_data:
             print("Error: Stock data is not available.")
             return None

        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        dividend_yield = stock_data['dividend_yield'] # Get yield
        expirations = stock_data.get('expirations', ())
        stock = stock_data['ticker_object']
        currency = stock_data.get('currency', 'USD')
        risk_free_rate = self.risk_free_rate # Use current rate

        # Handle missing volatility
        volatility_input = volatility # Store original/fetched volatility
        if volatility is None:
             print("\nWarning: Historical volatility unavailable. BSM prices/Greeks might be less accurate.")
             try:
                  user_vol = float(input("Enter an estimated annual volatility (e.g., 0.3) or Enter to use 0.3: ") or 0.3)
                  volatility = user_vol if user_vol > 0 else 1e-6
             except ValueError:
                  print("Invalid input. Using 0.3 volatility.")
                  volatility = 0.3
             print(f"Using estimated volatility for BSM/Greeks: {volatility*100:.2f}%")
        else:
            print(f"Using historical volatility for BSM/Greeks (if IV fails): {volatility*100:.2f}%")


        if risk_free_rate is None:
             risk_free_rate = self.get_risk_free_rate(verbose=True)

        # Select Expiration
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date: return None

        # Calculate time to expiration
        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days)
        T = days_to_expiration / 365.0
        print(f"Time to expiration (T): {T:.4f} years ({days_to_expiration} days)")

        # --- Fetch Option Chain Data ---
        try:
            print("\nFetching options chain data from yfinance...")
            # Use the caching helper
            self._chain_cache = {} # Clear previous cache before fetching new date
            options = stock.option_chain(expiration_date)
            calls = options.calls
            puts = options.puts
            self._chain_cache = {'date': expiration_date, 'calls': calls, 'puts': puts} # Store fetched data

            if calls.empty and puts.empty:
                 print(f"No options data found for {self.current_ticker} on {expiration_date}.")
                 return None

            print("Calculating BSM prices, IV, and Greeks...")

            # Combine calls and puts for easier processing
            calls = calls.add_prefix('call_')
            puts = puts.add_prefix('put_')
            calls = calls.rename(columns={'call_strike': 'strike'})
            puts = puts.rename(columns={'put_strike': 'strike'})
            # Select relevant columns before merge to avoid duplicate info
            call_cols = ['strike', 'call_lastPrice', 'call_bid', 'call_ask', 'call_volume', 'call_openInterest', 'call_impliedVolatility']
            put_cols = ['strike', 'put_lastPrice', 'put_bid', 'put_ask', 'put_volume', 'put_openInterest', 'put_impliedVolatility']
            # Keep only columns that exist in the fetched data
            call_cols = [col for col in call_cols if col in calls.columns]
            put_cols = [col for col in put_cols if col in puts.columns]

            chain_df = pd.merge(calls[call_cols], puts[put_cols], on='strike', how='outer')
            chain_df = chain_df.sort_values(by='strike').reset_index(drop=True)

            # --- Limit Strikes Around ATM ---
            max_strikes = self.config['max_strikes_chain']
            if len(chain_df) > max_strikes:
                atm_index = chain_df.iloc[(chain_df['strike'] - current_price).abs().argsort()[:1]].index[0]
                half_width = max_strikes // 2
                start_idx = max(0, atm_index - half_width)
                end_idx = min(len(chain_df), start_idx + max_strikes)
                if (end_idx - start_idx) < max_strikes: start_idx = max(0, end_idx - max_strikes) # Adjust if near edge
                chain_df = chain_df.iloc[start_idx:end_idx].reset_index(drop=True)
                print(f"Displaying {len(chain_df)} strikes around current price {self._format_currency(current_price, currency)}")

            # --- Calculate BSM, IV, Greeks ---
            results = []
            total_strikes = len(chain_df)
            for idx, row in chain_df.iterrows():
                print(f"\rProcessing strike {idx+1}/{total_strikes}...", end="") # Progress indicator

                strike = row['strike']
                data = {'strike': strike}

                # Market Prices (Use midpoint of bid/ask if lastPrice is unreliable or missing)
                call_bid = row.get('call_bid', 0)
                call_ask = row.get('call_ask', 0)
                market_call = row.get('call_lastPrice')
                if market_call is None or market_call <= 0 or (call_ask > 0 and market_call > call_ask) or (call_bid > 0 and market_call < call_bid):
                   market_call = (call_bid + call_ask) / 2 if call_bid > 0 and call_ask > 0 else None # Use midpoint if valid bid/ask

                put_bid = row.get('put_bid', 0)
                put_ask = row.get('put_ask', 0)
                market_put = row.get('put_lastPrice')
                if market_put is None or market_put <= 0 or (put_ask > 0 and market_put > put_ask) or (put_bid > 0 and market_put < put_bid):
                    market_put = (put_bid + put_ask) / 2 if put_bid > 0 and put_ask > 0 else None

                data['market_call'] = market_call
                data['market_put'] = market_put

                # Calculate IV (use calculated midpoint price)
                call_iv_calc = np.nan
                if pd.notna(market_call) and market_call > 0:
                    call_iv_calc = self.calculate_implied_volatility(current_price, strike, T, risk_free_rate, dividend_yield, market_call, "call")
                data['call_iv'] = call_iv_calc * 100 if pd.notna(call_iv_calc) else np.nan
                data['call_iv_yf'] = row.get('call_impliedVolatility', np.nan) * 100 # Store yf's IV too

                put_iv_calc = np.nan
                if pd.notna(market_put) and market_put > 0:
                    put_iv_calc = self.calculate_implied_volatility(current_price, strike, T, risk_free_rate, dividend_yield, market_put, "put")
                data['put_iv'] = put_iv_calc * 100 if pd.notna(put_iv_calc) else np.nan
                data['put_iv_yf'] = row.get('put_impliedVolatility', np.nan) * 100 # Store yf's IV too

                # Determine volatility to use for BSM/Greeks: Calculated IV > YF IV > Historical/Input Vol
                vol_to_use_call = call_iv_calc if pd.notna(call_iv_calc) else (row.get('call_impliedVolatility') if pd.notna(row.get('call_impliedVolatility')) else volatility)
                vol_to_use_put = put_iv_calc if pd.notna(put_iv_calc) else (row.get('put_impliedVolatility') if pd.notna(row.get('put_impliedVolatility')) else volatility)

                # Call Calculations
                data['bsm_call'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_call, "call")
                data['call_diff'] = market_call - data['bsm_call'] if pd.notna(market_call) and pd.notna(data['bsm_call']) else np.nan
                if self.config['show_greeks_in_chain']:
                     greeks_call = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_call, "call")
                     data.update({f'call_{k}': v for k, v in greeks_call.items()}) # Add greeks

                # Put Calculations
                data['bsm_put'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_put, "put")
                data['put_diff'] = market_put - data['bsm_put'] if pd.notna(market_put) and pd.notna(data['bsm_put']) else np.nan
                if self.config['show_greeks_in_chain']:
                     greeks_put = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, dividend_yield, vol_to_use_put, "put")
                     data.update({f'put_{k}': v for k, v in greeks_put.items()}) # Add greeks

                # Add other interesting data
                data['call_volume'] = row.get('call_volume', 0)
                data['call_oi'] = row.get('call_openInterest', 0)
                data['put_volume'] = row.get('put_volume', 0)
                data['put_oi'] = row.get('put_openInterest', 0)

                results.append(data)

            print("\r" + " " * 50 + "\rProcessing complete.") # Clear progress line

            results_df = pd.DataFrame(results)

             # --- Display Results ---
            # Define columns to display based on config and availability
            base_cols_c = ['call_volume', 'call_oi', 'market_call', 'bsm_call', 'call_iv']
            base_cols_p = ['put_iv', 'bsm_put', 'market_put', 'put_oi', 'put_volume']
            greek_cols_c = ['call_delta', 'call_gamma', 'call_theta', 'call_vega']
            greek_cols_p = ['put_delta', 'put_gamma', 'put_theta', 'put_vega']

            display_cols = base_cols_c + ['strike'] + base_cols_p
            if self.config['show_greeks_in_chain']:
                display_cols = base_cols_c + greek_cols_c + ['strike'] + greek_cols_p + base_cols_p

            # Filter df to only include desired columns that actually exist in results_df
            display_df = results_df[[col for col in display_cols if col in results_df.columns]].copy()

            # Rename columns for better display
            col_rename = {
                 'call_volume': 'C Vol', 'call_oi': 'C OI', 'market_call': 'C Market', 'bsm_call': 'C BSM', 'call_iv': 'C IV%', 'call_iv_yf': 'C IV%(YF)',
                 'call_delta': 'C Delta', 'call_gamma': 'C Gamma', 'call_theta': 'C Theta', 'call_vega': 'C Vega',
                 'strike': 'Strike',
                 'put_delta': 'P Delta', 'put_gamma': 'P Gamma', 'put_theta': 'P Theta', 'put_vega': 'P Vega',
                 'put_iv': 'P IV%', 'put_iv_yf': 'P IV%(YF)', 'bsm_put': 'P BSM', 'market_put': 'P Market', 'put_oi': 'P OI', 'put_volume': 'P Vol'
            }
            display_df.rename(columns=col_rename, inplace=True)

            # Custom formatting for the table (applied after rename)
            final_columns = display_df.columns
            float_format = ".2f" # Default for tabulate
            headers = [col_rename.get(col, col) for col in final_columns] # Use renamed headers

            # --- Print Table using Tabulate ---
            print(f"\n--- Options Chain for {self.current_ticker} ---")
            print(f"Expiration: {expiration_date} ({days_to_expiration} days)")
            print(f"Current Price: {self._format_currency(current_price, currency)}")
            print(f"Risk-Free Rate: {risk_free_rate*100:.2f}% | Div Yield: {dividend_yield*100:.2f}%")
            vol_source = "Hist/Est" if volatility_input is None else "Hist"
            print(f"BSM Volatility ({vol_source}): {volatility*100:.2f}% (used if IV unavailable)")
            print("-" * 100)

            # Use tabulate for printing - Apply formatting within tabulate if possible or format DF before
            # Formatting directly in DF for better control before tabulate
            formatters = {}
            for col in display_df.columns:
                if 'Market' in col or 'BSM' in col or 'Strike' in col or 'Theta' in col or 'Vega' in col:
                    formatters[col] = lambda x, c=currency: self._format_currency(x, c) if pd.notna(x) else 'N/A'
                elif 'IV%' in col:
                    formatters[col] = lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A'
                elif 'Delta' in col or 'Gamma' in col:
                     formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A'
                elif 'Vol' in col or 'OI' in col:
                    formatters[col] = lambda x: f"{int(x):,}" if pd.notna(x) else ('0' if x==0 else 'N/A')

            # Apply formatting - Create a display copy
            display_df_formatted = display_df.copy()
            for col, func in formatters.items():
                 if col in display_df_formatted.columns:
                     display_df_formatted[col] = display_df_formatted[col].apply(func)


            # Print with tabulate
            print(tabulate(display_df_formatted, headers='keys', tablefmt='pretty', # 'pretty' or 'psql'
                           showindex=False, numalign="right", stralign="right"))
            print("-" * 100)


            # Ask to visualize if requested
            if visualize:
                 print("Generating visualization...")
                 self.visualize_options_chain(results_df, current_price, currency, expiration_date) # Pass raw results_df
            else:
                visualize_input = input("\nVisualize this options chain (Price/IV)? (y/n): ").lower()
                if visualize_input == 'y':
                     print("Generating visualization...")
                     self.visualize_options_chain(results_df, current_price, currency, expiration_date)

            return results_df # Return the raw calculated data

        except AttributeError as ae:
             # This specific error often means options aren't available for the date/ticker
             print(f"\nError: Options data structure not found for {self.current_ticker} on {expiration_date}. Ticker might not have options or data is temporarily unavailable.")
             if self.config['debug_mode']: traceback.print_exc()
             return None
        except Exception as e:
            print(f"\nError calculating options chain: {e}")
            if self.config['debug_mode']:
                traceback.print_exc()
            return None

    def visualize_options_chain(self, df, current_price, currency, expiration_date):
        """Visualize the options chain data. Uses matplotlib."""
        if df is None or df.empty:
            print("No data available to visualize.")
            return
        if 'strike' not in df.columns:
            print("Error: DataFrame missing 'strike' column for visualization.")
            return

        df = df.copy().dropna(subset=['strike']) # Work on a copy, ensure strikes are valid
        if df.empty:
             print("No valid strike data to visualize after dropping NaNs.")
             return

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True) # Slightly adjusted size
        fig.suptitle(f"{self.current_ticker} Options Chain ({expiration_date})\nCurrent Price: {self._format_currency(current_price, currency)}", fontsize=15, weight='bold')
        fig.patch.set_facecolor('white') # Set background color

        # --- Plot 1: Prices (Market vs. BSM) ---
        ax1 = axes[0]
        ax1.set_facecolor('#f0f0f0') # Light grey background for plot area
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{self._format_currency(1, currency)[0]}%.2f')) # Format Y axis as currency

        # Calls
        if 'market_call' in df.columns and df['market_call'].notna().any():
            ax1.plot(df['strike'], df['market_call'], marker='o', linestyle='-', color='blue', markersize=5, alpha=0.8, label='Market Call')
        if 'bsm_call' in df.columns and df['bsm_call'].notna().any():
            ax1.plot(df['strike'], df['bsm_call'], linestyle='--', color='cyan', alpha=0.8, label='BSM Call')
        # Puts
        if 'market_put' in df.columns and df['market_put'].notna().any():
             ax1.plot(df['strike'], df['market_put'], marker='o', linestyle='-', color='red', markersize=5, alpha=0.8, label='Market Put')
        if 'bsm_put' in df.columns and df['bsm_put'].notna().any():
             ax1.plot(df['strike'], df['bsm_put'], linestyle='--', color='magenta', alpha=0.8, label='BSM Put')

        ax1.set_ylabel(f'Option Price ({currency})', fontsize=11, weight='medium')
        ax1.set_title('Market vs. Calculated (BSM) Prices', fontsize=13, weight='medium')
        ax1.grid(True, linestyle=':', linewidth=0.5, color='grey')
        # Add vertical line for current price
        ax1.axvline(current_price, color='black', linestyle=':', lw=1.5, label=f'Current Price')
        ax1.legend(fontsize=9, loc='best')
        ax1.tick_params(axis='both', which='major', labelsize=10)

        # --- Plot 2: Implied Volatility ---
        ax2 = axes[1]
        ax2.set_facecolor('#f0f0f0')
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%')) # Format Y axis as percentage

        # Plot calculated IV first if available
        if 'call_iv' in df.columns and df['call_iv'].notna().any():
            ax2.plot(df['strike'], df['call_iv'], marker='^', linestyle='-', color='green', markersize=5, label='Call IV (Calculated)')
        if 'put_iv' in df.columns and df['put_iv'].notna().any():
            ax2.plot(df['strike'], df['put_iv'], marker='v', linestyle='-', color='orange', markersize=5, label='Put IV (Calculated)')

        # Plot YFinance IV as dashed lines for comparison (if present)
        if 'call_iv_yf' in df.columns and df['call_iv_yf'].notna().any():
            ax2.plot(df['strike'], df['call_iv_yf'], linestyle=':', color='lightgreen', markersize=4, label='Call IV (YF)')
        if 'put_iv_yf' in df.columns and df['put_iv_yf'].notna().any():
            ax2.plot(df['strike'], df['put_iv_yf'], linestyle=':', color='yellow', markersize=4, label='Put IV (YF)')

        # Add horizontal line for historical volatility if available
        if self.current_stock_data and self.current_stock_data['volatility'] is not None:
             hist_vol = self.current_stock_data['volatility'] * 100
             ax2.axhline(hist_vol, color='dimgray', linestyle='--', lw=1.5, label=f'Hist. Vol ({hist_vol:.1f}%)')

        ax2.set_xlabel('Strike Price', fontsize=11, weight='medium')
        ax2.set_ylabel('Implied Volatility (%)', fontsize=11, weight='medium')
        ax2.set_title('Implied Volatility Smile / Skew', fontsize=13, weight='medium')
        ax2.grid(True, linestyle=':', linewidth=0.5, color='grey')
        # Add vertical line for current price
        ax2.axvline(current_price, color='black', linestyle=':', lw=1.5)
        ax2.legend(fontsize=9, loc='best')
        ax2.tick_params(axis='both', which='major', labelsize=10)

        # Adjust y-axis limits for IV plot for better visibility
        valid_ivs = df[['call_iv', 'put_iv', 'call_iv_yf', 'put_iv_yf']].unstack().dropna()
        if not valid_ivs.empty:
            min_iv = valid_ivs.min()
            max_iv = valid_ivs.max()
            padding = (max_iv - min_iv) * 0.1
            ax2.set_ylim(max(0, min_iv - padding), max_iv + padding)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        try:
             plt.show()
        except Exception as e:
             print(f"\nError displaying plot: {e}. Ensure a GUI backend is available.")
             # Consider saving the plot as a fallback:
             # save_path = f"{self.current_ticker}_{expiration_date}_chain.png"
             # try:
             #     fig.savefig(save_path)
             #     print(f"Plot saved to {save_path}")
             # except Exception as save_err:
             #     print(f"Could not save plot: {save_err}")


    # --- Options Strategy Analysis ---

    def _calculate_payoff(self, S_T, strategy_legs):
        """Calculates the Profit/Loss of a strategy at expiration price S_T."""
        total_payoff = 0
        # Calculate net initial cost/credit first
        net_cost = 0
        for leg in strategy_legs:
            if leg['dir'] == 'long': net_cost += leg['price']
            else: net_cost -= leg['price'] # Subtract credit received

        # Calculate terminal value of each leg
        for leg in strategy_legs:
            leg_type, direction, K = leg['type'], leg['dir'], leg.get('K')
            payoff_leg = 0
            if leg_type == 'stock':
                payoff_leg = S_T # Payoff is the stock price itself
            elif leg_type == 'call':
                payoff_leg = max(0, S_T - K) if K is not None else 0
            elif leg_type == 'put':
                payoff_leg = max(0, K - S_T) if K is not None else 0

            # If short, the payoff is negative
            if direction == 'short':
                payoff_leg *= -1

            total_payoff += payoff_leg

        # Profit/Loss = Terminal Value of Position - Net Initial Cost
        profit_loss = total_payoff - net_cost
        return profit_loss

    def _plot_payoff(self, S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency):
        """Plots the Profit/Loss diagram for a strategy."""
        fig, ax = plt.subplots(figsize=(11, 6.5)) # Slightly larger figure
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f0f0f0')
        ax.plot(S_T_range, PnL, lw=2.5, color='navy', label='Profit/Loss at Expiration')

        # Format axes
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{self._format_currency(1, currency)[0]}%.2f'))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'{self._format_currency(1, currency)[0]}%.2f'))
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Mark zero profit line
        ax.axhline(0, color='black', linestyle='--', lw=1, label='Breakeven Level')

        # Mark breakeven points clearly
        valid_bes = [be for be in breakevens if pd.notna(be)] # Filter out NaNs
        if valid_bes:
            ax.scatter(valid_bes, [0] * len(valid_bes), color='red', s=100, zorder=5, edgecolors='black', label='Breakeven(s)')
            for be in valid_bes:
                 # Add text annotation slightly above the point
                 ax.text(be, max(abs(PnL.min()), abs(PnL.max())) * 0.05, # Adjust vertical offset dynamically
                         f'BE: {self._format_currency(be, currency)}',
                         color='darkred', ha='center', va='bottom', fontsize=9, weight='bold')

        # Annotate Max Profit / Max Loss on plot edges if defined
        profit_label = f'Max Profit: {self._format_currency(max_profit, currency)}' if (pd.notna(max_profit) and max_profit != float('inf')) else 'Max Profit: Unlimited'
        loss_label = f'Max Loss: {self._format_currency(max_loss, currency)}' if (pd.notna(max_loss) and max_loss != float('-inf')) else 'Max Loss: Unlimited' # Note: Max loss for long stock is not unlimited

        if pd.notna(max_profit) and max_profit != float('inf'):
            ax.axhline(max_profit, color='green', linestyle=':', lw=1.5, label=profit_label)
        elif max_profit == float('inf'):
             # Indicate unlimited profit visually if possible (e.g., arrow?) - or just text
              ax.text(S_T_range[-1], PnL[-1], ' Unlimited Profit ->', color='green', ha='right', va='center', fontsize=10, weight='bold')


        if pd.notna(max_loss) and max_loss != float('-inf'):
             ax.axhline(max_loss, color='red', linestyle=':', lw=1.5, label=loss_label)
        elif max_loss == float('-inf'):
             # Indicate unlimited loss (typical for short positions without cover)
             ax.text(S_T_range[0], PnL[0], '<- Unlimited Loss ', color='red', ha='left', va='center', fontsize=10, weight='bold')


        ax.set_title(f'{strategy_name} Payoff Diagram', fontsize=15, weight='bold')
        ax.set_xlabel(f'Underlying Price at Expiration ({currency})', fontsize=11, weight='medium')
        ax.set_ylabel(f'Profit / Loss ({currency})', fontsize=11, weight='medium')
        ax.grid(True, linestyle=':', linewidth=0.5, color='grey')

        # Dynamic Y-axis limits based on calculated PnL and defined max/min
        y_min_plot = PnL.min()
        y_max_plot = PnL.max()
        if pd.notna(max_loss) and max_loss != float('-inf'): y_min_plot = min(y_min_plot, max_loss)
        if pd.notna(max_profit) and max_profit != float('inf'): y_max_plot = max(y_max_plot, max_profit)

        padding = (y_max_plot - y_min_plot) * 0.1 # 10% padding
        ax.set_ylim(y_min_plot - padding, y_max_plot + padding)

        ax.legend(fontsize=9, loc='best')
        plt.tight_layout()

        try:
             plt.show()
        except Exception as e:
             print(f"\nError displaying plot: {e}.")


    def analyze_strategy(self):
        """Guides user through selecting and analyzing an options strategy (Console)."""
        if self.current_stock_data is None:
             print("\nPlease fetch stock data first (Option 1).")
             ticker = input("Enter stock ticker symbol: ").upper()
             if not ticker or not self.get_stock_data(ticker): return
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change_ticker = input("Analyze strategy for a different ticker? (y/n, default n): ").lower()
             if change_ticker == 'y':
                  ticker = input("Enter new stock ticker symbol: ").upper()
                  if not ticker or not self.get_stock_data(ticker): return

        stock_data = self.current_stock_data
        S0 = stock_data['current_price'] # Price at time of analysis
        expirations = stock_data['expirations']
        currency = stock_data['currency']
        dividend_yield = stock_data['dividend_yield'] # Get yield for potential BSM estimate
        risk_free_rate = self.risk_free_rate
        volatility = stock_data['volatility'] # Base volatility for BSM estimate

        # --- Strategy Selection ---
        print("\n--- Options Strategy Analysis ---")
        print("Select a strategy:")
        print(" 1. Covered Call (Long Stock + Short Call)")
        print(" 2. Protective Put (Long Stock + Long Put)")
        print(" 3. Bull Call Spread (Long Lower Call + Short Higher Call)")
        print(" 4. Bear Put Spread (Long Higher Put + Short Lower Put)")
        print(" 5. Long Straddle (Long ATM Call + Long ATM Put)")
        print(" 6. Long Strangle (Long OTM Call + Long OTM Put)")
        # Add more strategies here...
        print(" 0. Back to Main Menu")

        strategy_choice = None
        while strategy_choice is None:
            try:
                choice = input("Enter strategy number: ")
                strategy_choice = int(choice)
                if not (0 <= strategy_choice <= 6): # Adjust range if more strategies added
                    print("Invalid choice.")
                    strategy_choice = None
            except ValueError:
                print("Invalid input. Please enter a number.")

        if strategy_choice == 0: return

        # --- Select Expiration ---
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date: return

        # Helper function to get option price (market or BSM estimate)
        def get_option_price(strike, option_type, exp_date):
            option_data = self._get_option_data_for_strike(exp_date, strike, option_type)
            market_price = None
            if option_data is not None:
                # Try midpoint first
                bid = option_data.get('bid', 0)
                ask = option_data.get('ask', 0)
                last = option_data.get('lastPrice')
                if bid > 0 and ask > 0:
                    market_price = (bid + ask) / 2
                    # Sanity check midpoint vs last price if available
                    if last and (market_price < last * 0.8 or market_price > last * 1.2):
                        print(f" Note: Midpoint ({self._format_currency(market_price, currency)}) differs significantly from Last ({self._format_currency(last, currency)}) for {option_type} K={strike}. Using midpoint.")
                elif last and last > 0:
                    market_price = last # Fallback to last price
                    print(f" Note: Using last traded price for {option_type} K={strike} as bid/ask is missing/zero.")


            if market_price is not None and market_price > 0:
                print(f"Using market price ({self._format_currency(market_price, currency)}) for {option_type.capitalize()} K={strike}.")
                return market_price
            else:
                # Fallback to BSM estimate
                print(f"Warning: Market price for {option_type.capitalize()} K={strike} unavailable