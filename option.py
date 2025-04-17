# options_analyzer.py
# Enhanced Options Pricing and Analysis Tool

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
    common option strategies with payoff diagrams.
    """
    def __init__(self):
        """Initialize the Options Analyzer with default parameters"""
        self.current_ticker = None
        self.current_stock_data = None
        self.risk_free_rate = None
        self.config = self._load_config()
        self.favorite_tickers = self._load_favorite_tickers()
        # Fetch initial risk-free rate
        self.get_risk_free_rate()

    # --- Configuration and Persistence ---

    def _load_config(self):
        """Load configuration from file or use defaults."""
        default_config = {
            'volatility_days': 252,  # Trading days for annualization
            'default_risk_free_rate': 0.04,  # 4% fallback
            'show_greeks_in_chain': True,
            'max_strikes_chain': 15,  # Max strikes around ATM in chain display
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
            if currency == 'USD':
                return f"${value:,.2f}"
            # Add other common currency symbols or formats here if needed
            else:
                return f"{value:,.2f} {currency}"
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
            # Fetching a small piece of info is faster than history
            info = stock.info
            if not info or info.get('quoteType') == 'MUTUALFUND': # Exclude mutual funds for options
                 # Check history as a fallback for less common tickers
                 hist = stock.history(period="5d")
                 if hist.empty:
                     print(f"Ticker '{ticker}' is not valid or no recent data available.")
                     return False
            # Basic check if it's likely an equity or ETF
            if info.get('quoteType') not in ['EQUITY', 'ETF']:
                 print(f"Warning: Ticker '{ticker}' may not be an equity or ETF ({info.get('quoteType')}). Options might not be available.")
                 # Allow proceeding but warn the user

            print(f"Ticker '{ticker}' appears valid ({info.get('shortName', 'N/A')}).")
            return True
        except Exception as e:
            print(f"Error validating ticker '{ticker}': {e}")
            if self.config['debug_mode']:
                import traceback
                traceback.print_exc()
            return False

    def _select_expiration_date(self, expirations):
        """Lists available expiration dates and prompts user selection."""
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
                selection = input(f"\nSelect expiration date (1-{len(valid_expirations)}): ")
                if not selection: # Handle empty input
                    print("Using first available date.")
                    selected_exp = valid_expirations[0]
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
        """Fetch stock data, company info, and options expirations using yfinance."""
        ticker = ticker.upper().strip()
        if not self.validate_ticker(ticker):
            return None

        try:
            print(f"\nFetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            # Get last year of history for volatility calculation
            hist = stock.history(period="1y")

            if hist.empty:
                print(f"Could not fetch historical data for {ticker}.")
                # Try fetching current quote info as fallback for price
                info = stock.info
                current_price = info.get('currentPrice') or info.get('previousClose')
                if not current_price:
                     raise ValueError(f"Could not fetch current price for {ticker}")
                volatility = None # Cannot calculate historical volatility
                print("Warning: Using current/previous close price, volatility calculation skipped.")
            else:
                current_price = hist['Close'].iloc[-1]
                # Calculate historical volatility (annualized)
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                if len(returns) < 2: # Need at least 2 returns for std dev
                     volatility = None
                     print("Warning: Not enough historical data to calculate volatility.")
                else:
                     volatility = returns.std() * np.sqrt(self.config['volatility_days'])


            # Get company info
            company_name = ticker
            sector = 'N/A'
            industry = 'N/A'
            market_cap_str = 'N/A'
            currency = 'USD' # Default
            try:
                info = stock.info
                company_name = info.get('shortName', ticker)
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                market_cap = info.get('marketCap')
                currency = info.get('currency', 'USD')
                if market_cap:
                    if market_cap >= 1e12:
                        market_cap_str = f"{self._format_currency(market_cap / 1e12, currency)}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"{self._format_currency(market_cap / 1e9, currency)}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"{self._format_currency(market_cap / 1e6, currency)}M"
                    else:
                         market_cap_str = self._format_currency(market_cap, currency)

            except Exception as e:
                if self.config['debug_mode']:
                    print(f"Could not fetch detailed company info: {e}")

            # Print stock details
            print(f"\n=== {company_name} ({ticker}) ===")
            print(f"Current price: {self._format_currency(current_price, currency)}")
            print(f"Sector: {sector}")
            print(f"Industry: {industry}")
            print(f"Market Cap: {market_cap_str}")
            if volatility is not None:
                print(f"Annualized Volatility (1y): {volatility:.4f} ({volatility*100:.2f}%)")
            else:
                 print("Annualized Volatility (1y): N/A")

            # Get available expiration dates
            expirations = ()
            try:
                 expirations = stock.options
                 if not expirations:
                      print("Note: No options expiration dates found for this ticker.")
            except Exception as e:
                 print(f"Could not fetch options expiration dates: {e}")


            self.current_ticker = ticker
            self.current_stock_data = {
                'ticker': ticker,
                'current_price': current_price,
                'volatility': volatility, # Can be None
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
                import traceback
                traceback.print_exc()
            # Reset current data if fetch fails
            if self.current_ticker == ticker:
                 self.current_ticker = None
                 self.current_stock_data = None
            return None

    def get_risk_free_rate(self):
        """Get risk-free rate from Treasury yield (10-year as proxy) or default."""
        try:
            # print("Fetching current risk-free rate (10-Year Treasury Yield)...")
            # Using ^TNX for 10-Year Treasury Yield Index CBOE
            treasury = yf.Ticker("^TNX")
            # Fetch only the most recent data point needed
            data = treasury.history(period="5d") # Fetch a few days for robustness
            if not data.empty:
                # Use the latest closing value, convert from percentage to decimal
                rate = data['Close'].iloc[-1] / 100
                # Basic sanity check for the rate
                if 0 <= rate <= 0.2: # Assume rate won't be negative or > 20%
                    print(f"Using current risk-free rate (10Y Treasury): {rate:.4f} ({rate*100:.2f}%)")
                    self.risk_free_rate = rate
                    return rate
                else:
                     print(f"Warning: Fetched treasury rate ({rate:.4f}) seems unusual. Falling back to default.")
            else:
                 print("Could not fetch treasury data. Falling back to default.")

        except Exception as e:
            print(f"Error fetching risk-free rate: {e}. Falling back to default.")
            if self.config['debug_mode']:
                import traceback
                traceback.print_exc()

        # Fallback to default if API fails or data is unusual
        default_rate = self.config['default_risk_free_rate']
        print(f"Using default risk-free rate: {default_rate:.4f} ({default_rate*100:.2f}%)")
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
             opt_chain = stock.option_chain(expiration_date)
             data = opt_chain.calls if option_type == 'call' else opt_chain.puts

             # Find the specific strike row
             option_data = data[data['strike'] == strike]

             if option_data.empty:
                 print(f"Warning: No {option_type} data found for strike {strike} on {expiration_date}.")
                 # Could try finding the closest strike as a fallback here if desired
                 return None

             # Return the first row if multiple contracts match (shouldn't happen for std options)
             return option_data.iloc[0]

         except IndexError: # Handle cases where yfinance might return unexpected empty structures
              print(f"Error: No options data available for {self.current_ticker} on {expiration_date}.")
              return None
         except Exception as e:
             print(f"Error fetching option data for strike {strike} ({option_type}): {e}")
             if self.config['debug_mode']:
                 import traceback
                 traceback.print_exc()
             return None

    # --- Black-Scholes-Merton Model and Greeks ---

    def black_scholes_merton(self, S, K, T, r, sigma, option_type="call"):
        """
        Calculate option price using Black-Scholes-Merton model.

        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility of the stock (annualized)
        option_type: "call" or "put"

        Returns: Option price or np.nan if inputs are invalid.
        """
        # Input validation
        if T < 0: T = 0 # Handle expired options (intrinsic value)
        if sigma <= 0: sigma = 1e-6 # Prevent division by zero, use tiny volatility
        if S <=0 or K <= 0: return np.nan # Prices must be positive

        # Handle immediate expiration
        if T == 0:
            if option_type.lower() == "call":
                return max(0.0, S - K)
            elif option_type.lower() == "put":
                return max(0.0, K - S)
            else:
                return np.nan # Invalid option type

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        try:
            if option_type.lower() == "call":
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == "put":
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                print(f"Warning: Invalid option type '{option_type}' in BSM.")
                return np.nan
            return max(0.0, price) # Price cannot be negative
        except OverflowError:
            print("Warning: Overflow encountered in BSM calculation. Inputs might be extreme.")
            return np.nan
        except Exception as e:
            print(f"Error in BSM calculation: {e}")
            return np.nan

    def calculate_option_greeks(self, S, K, T, r, sigma, option_type="call"):
        """
        Calculate option Greeks using Black-Scholes-Merton model.

        Returns: Dictionary with Delta, Gamma, Theta, Vega, Rho or None if inputs invalid.
        """
        greeks = { "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan }
        # Input validation
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            # For expired options, some greeks might be definable, but returning NaN is safer.
            # Delta might be 1 or 0 for calls, -1 or 0 for puts. Others usually 0.
            # We return NaNs to indicate the model doesn't apply well at expiration.
            return greeks

        option_type = option_type.lower()

        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            sqrt_T = np.sqrt(T)
            pdf_d1 = norm.pdf(d1)

            # Gamma - rate of change of Delta (same for calls and puts)
            greeks["gamma"] = pdf_d1 / (S * sigma * sqrt_T) if (S * sigma * sqrt_T) != 0 else 0

            # Vega - sensitivity to volatility change (per 1% change)
            greeks["vega"] = (S * sqrt_T * pdf_d1) / 100

            # Delta - sensitivity to underlying price change
            if option_type == "call":
                greeks["delta"] = norm.cdf(d1)
                # Theta - sensitivity to time decay (per day)
                term1 = - (S * pdf_d1 * sigma) / (2 * sqrt_T)
                term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
                greeks["theta"] = (term1 + term2) / 365  # Convert annual theta to daily
                # Rho - sensitivity to interest rate change (per 1% change)
                greeks["rho"] = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
            elif option_type == "put":
                greeks["delta"] = norm.cdf(d1) - 1
                # Theta
                term1 = - (S * pdf_d1 * sigma) / (2 * sqrt_T)
                term2 = + r * K * np.exp(-r * T) * norm.cdf(-d2) # Note the '+' sign
                greeks["theta"] = (term1 + term2) / 365 # Convert annual theta to daily
                # Rho
                greeks["rho"] = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
            else:
                print(f"Warning: Invalid option type '{option_type}' for Greeks.")
                return {k: np.nan for k in greeks}

            return greeks

        except ZeroDivisionError:
            print("Warning: Division by zero encountered in Greeks calculation (likely T or sigma is zero).")
            return {k: 0.0 for k in greeks} # Return zeros in this edge case
        except OverflowError:
            print("Warning: Overflow encountered in Greeks calculation. Inputs might be extreme.")
            return {k: np.nan for k in greeks}
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            return {k: np.nan for k in greeks}

    def calculate_implied_volatility(self, S, K, T, r, market_price, option_type="call"):
        """
        Calculate implied volatility using a bisection method.

        Returns: Implied volatility (annualized decimal) or np.nan if calculation fails.
        """
        option_type = option_type.lower()
        precision = self.config['iv_precision']
        max_iterations = self.config['iv_max_iterations']

        # --- Input Validation and Edge Cases ---
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return np.nan # Cannot calculate IV for zero price or expired/invalid options

        # Check if market price is below intrinsic value (arbitrage or bad data)
        intrinsic_value = 0.0
        if option_type == "call":
            intrinsic_value = max(0.0, S - K * np.exp(-r * T)) # Use discounted strike for comparison
        elif option_type == "put":
            intrinsic_value = max(0.0, K * np.exp(-r * T) - S)
        else:
             print(f"Warning: Invalid option type '{option_type}' for IV.")
             return np.nan

        # Allow a small tolerance for rounding errors in market data
        if market_price < intrinsic_value - precision:
            # print(f"Warning: Market price ({market_price:.4f}) is below intrinsic value ({intrinsic_value:.4f}) for {option_type} K={K}. Cannot calculate IV.")
            return np.nan

        # --- Bisection Method ---
        vol_low = 1e-5  # Lower bound for volatility (slightly above zero)
        vol_high = 5.0  # Upper bound (500% volatility) - adjust if needed

        # Calculate price at bounds
        price_low = self.black_scholes_merton(S, K, T, r, vol_low, option_type)
        price_high = self.black_scholes_merton(S, K, T, r, vol_high, option_type)

        # Check if market price is outside the possible range for vol_low/vol_high
        if pd.isna(price_low) or pd.isna(price_high):
             # print(f"Warning: BSM returned NaN at volatility bounds for K={K}. Cannot calculate IV.")
             return np.nan

        if market_price <= price_low:
            return vol_low # Market price is below price at minimum volatility
        if market_price >= price_high:
            return vol_high # Market price is above price at maximum volatility

        # Bisection search loop
        for _ in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2
            price_mid = self.black_scholes_merton(S, K, T, r, vol_mid, option_type)

            if pd.isna(price_mid): # Handle BSM errors during iteration
                 # Try adjusting bounds slightly, could be numerical instability
                 if self.config['debug_mode']: print(f"BSM returned NaN at vol_mid={vol_mid}. Adjusting IV bounds.")
                 vol_high = vol_mid # Or vol_low = vol_mid, depending on direction needed
                 continue

            # Check for convergence
            if abs(price_mid - market_price) < precision:
                return vol_mid

            # Narrow the search interval
            if price_mid > market_price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid

            # Check if bounds are too close
            if abs(vol_high - vol_low) < precision:
                 break


        # If max_iterations reached without convergence, return the midpoint
        final_vol = (vol_low + vol_high) / 2
        # Final check: is the price at final_vol reasonably close?
        final_price = self.black_scholes_merton(S, K, T, r, final_vol, option_type)
        if pd.notna(final_price) and abs(final_price - market_price) < precision * 10: # Looser check
             return final_vol
        else:
             # print(f"Warning: IV calculation did not converge sufficiently for K={K}. Market: {market_price:.2f}, Model: {final_price:.2f} at IV {final_vol*100:.2f}%")
             return np.nan # Indicate failure to converge reliably

    # --- Core Functionality Methods ---

    def get_simple_option_price(self):
        """Calculate and display a simple option price based on user input."""
        if self.current_stock_data is None:
            print("\nPlease fetch stock data first (Option 1).")
            ticker = input("Enter stock ticker symbol: ").upper()
            if not ticker: return
            if not self.get_stock_data(ticker):
                return # Exit if fetching fails
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change_ticker = input("Fetch data for a different ticker? (y/n, default n): ").lower()
             if change_ticker == 'y':
                  ticker = input("Enter new stock ticker symbol: ").upper()
                  if not ticker: return
                  if not self.get_stock_data(ticker):
                       return # Exit if fetching fails

        # Use currently loaded data
        stock_data = self.current_stock_data
        if not stock_data: # Should not happen if logic above is correct, but safety check
             print("Error: Stock data is not available.")
             return

        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        expirations = stock_data['expirations']
        stock = stock_data['ticker_object']
        currency = stock_data['currency']

        if volatility is None:
             print("Warning: Historical volatility is not available. BSM prices may be inaccurate.")
             # Optionally, prompt user for a volatility estimate
             try:
                  user_vol = float(input("Enter an estimated annual volatility (e.g., 0.3 for 30%) or press Enter to use 0: "))
                  volatility = user_vol if user_vol > 0 else 1e-6 # Use tiny vol if 0 or negative
             except ValueError:
                  print("Invalid input. Using 0 volatility.")
                  volatility = 1e-6 # Use tiny vol if input error


        if self.risk_free_rate is None: # Should have been fetched at init, but check again
             risk_free_rate = self.get_risk_free_rate()
        else:
             risk_free_rate = self.risk_free_rate

        # --- Select Expiration ---
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date:
            return # User didn't select or no dates available

        # Calculate time to expiration
        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days) # Ensure non-negative
        T = days_to_expiration / 365.0 # Use 365 for consistency
        print(f"Time to expiration (T): {T:.4f} years")

        # --- Select Strike Price ---
        strike = None
        while strike is None:
            strike_input = input(f"\nEnter strike price (e.g., {current_price:.2f}) or 'atm' for closest: ").lower()
            if strike_input == 'atm':
                try:
                    options = stock.option_chain(expiration_date)
                    # Combine call and put strikes, find unique ones, sort
                    all_strikes = sorted(list(set(options.calls['strike'].tolist() + options.puts['strike'].tolist())))
                    if not all_strikes:
                         print("No strikes found for this expiration. Using current price.")
                         strike = current_price
                    else:
                         # Find the strike closest to the current price
                         strike = min(all_strikes, key=lambda x: abs(x - current_price))
                         print(f"Found closest available strike: {self._format_currency(strike, currency)}")
                except Exception as e:
                    print(f"Could not fetch available strikes. Using current price. Error: {e}")
                    strike = current_price # Fallback if fetching strikes fails
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
            option_type_input = input("Calculate for 'call', 'put', or 'both'? (default 'both'): ").lower()
            if not option_type_input:
                option_type = 'both'
            elif option_type_input in ['call', 'put', 'both']:
                 option_type = option_type_input
            else:
                 print("Invalid option type.")

        # --- Calculate and Display ---
        results = {}
        if option_type in ['call', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "call")
            greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, volatility, "call")
            results['call'] = {'price': bsm_price, 'greeks': greeks}

        if option_type in ['put', 'both']:
            bsm_price = self.black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "put")
            greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, volatility, "put")
            results['put'] = {'price': bsm_price, 'greeks': greeks}

        print(f"\n--- BSM Option Analysis ---")
        print(f"Stock: {self.current_ticker} @ {self._format_currency(current_price, currency)}")
        print(f"Strike: {self._format_currency(strike, currency)}")
        print(f"Expiration: {expiration_date} ({days_to_expiration} days)")
        print(f"Volatility (Input): {volatility*100:.2f}%")
        print(f"Risk-Free Rate: {risk_free_rate*100:.2f}%")
        print("-" * 25)

        if 'call' in results:
            print(f"BSM Call Price: {self._format_currency(results['call']['price'], currency)}")
            if results['call']['greeks']:
                g = results['call']['greeks']
                print("  Greeks:")
                print(f"    Delta: {g['delta']:.4f}")
                print(f"    Gamma: {g['gamma']:.4f}")
                print(f"    Theta: {self._format_currency(g['theta'], currency)} / day")
                print(f"    Vega:  {self._format_currency(g['vega'], currency)} / 1% vol")
                print(f"    Rho:   {self._format_currency(g['rho'], currency)} / 1% rate")
            print("-" * 25)


        if 'put' in results:
            print(f"BSM Put Price: {self._format_currency(results['put']['price'], currency)}")
            if results['put']['greeks']:
                g = results['put']['greeks']
                print("  Greeks:")
                print(f"    Delta: {g['delta']:.4f}")
                print(f"    Gamma: {g['gamma']:.4f}")
                print(f"    Theta: {self._format_currency(g['theta'], currency)} / day")
                print(f"    Vega:  {self._format_currency(g['vega'], currency)} / 1% vol")
                print(f"    Rho:   {self._format_currency(g['rho'], currency)} / 1% rate")
            print("-" * 25)

    def calculate_options_chain(self):
        
        
            if specific_expiration: # Use passed expiration if provided
                expiration_date = specific_expiration
                if expiration_date not in expirations:
                    print(f"Error: Provided expiration '{expiration_date}' not valid for {ticker}.")
                    return None
                print(f"\nUsing specified expiration date: {expiration_date}")
            else: # Otherwise, prompt the user (for console use) or select first (better for API use)
                # For streamlit, we rely on specific_expiration being passed.
                # If called without it from backend, maybe default to first or raise error.
                # Let's keep the selection logic but know it won't be hit from app.py
                expiration_date = self._select_expiration_date(expirations)
                if not expiration_date:
                    return None
        """Calculate and display a detailed options chain for a selected expiration."""
        if self.current_stock_data is None:
            print("\nPlease fetch stock data first (Option 1).")
            ticker = input("Enter stock ticker symbol: ").upper()
            if not ticker: return
            if not self.get_stock_data(ticker):
                return # Exit if fetching fails
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change_ticker = input("Fetch data for a different ticker? (y/n, default n): ").lower()
             if change_ticker == 'y':
                  ticker = input("Enter new stock ticker symbol: ").upper()
                  if not ticker: return
                  if not self.get_stock_data(ticker):
                       return # Exit if fetching fails

        # Use currently loaded data
        stock_data = self.current_stock_data
        if not stock_data:
             print("Error: Stock data is not available.")
             return

        current_price = stock_data['current_price']
        volatility = stock_data['volatility']
        expirations = stock_data['expirations']
        stock = stock_data['ticker_object']
        currency = stock_data['currency']

        if volatility is None:
             print("\nWarning: Historical volatility is not available. BSM prices will be based on zero volatility unless you provide one.")
             try:
                  user_vol = float(input("Enter an estimated annual volatility (e.g., 0.3 for 30%) or press Enter to use 0: "))
                  volatility = user_vol if user_vol > 0 else 1e-6
             except ValueError:
                  print("Invalid input. Using 0 volatility.")
                  volatility = 1e-6

        if self.risk_free_rate is None:
             risk_free_rate = self.get_risk_free_rate()
        else:
             risk_free_rate = self.risk_free_rate

        # Select Expiration
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date:
            return

        # Calculate time to expiration
        today = dt.datetime.now().date()
        exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
        days_to_expiration = max(0, (exp_date - today).days)
        T = days_to_expiration / 365.0
        print(f"Time to expiration (T): {T:.4f} years")

        # --- Fetch Option Chain Data ---
        try:
            print("\nFetching options chain data from yfinance...")
            options = stock.option_chain(expiration_date)
            calls = options.calls
            puts = options.puts

            if calls.empty and puts.empty:
                 print(f"No options data found for {self.current_ticker} on {expiration_date}.")
                 return None # Indicate no data found

            print("Calculating BSM prices, IV, and Greeks...")

            # Combine calls and puts for easier processing
            calls = calls.add_prefix('call_')
            puts = puts.add_prefix('put_')
            # Ensure strike columns have same name for merging
            calls = calls.rename(columns={'call_strike': 'strike'})
            puts = puts.rename(columns={'put_strike': 'strike'})

            # Merge call and put data on strike price
            # Use outer merge to keep all strikes even if one side is missing
            chain_df = pd.merge(calls, puts, on='strike', how='outer')
            chain_df = chain_df.sort_values(by='strike').reset_index(drop=True)

            # --- Limit Strikes Around ATM ---
            max_strikes = self.config['max_strikes_chain']
            if len(chain_df) > max_strikes:
                atm_index = chain_df.iloc[(chain_df['strike'] - current_price).abs().argsort()[:1]].index[0]
                half_width = max_strikes // 2
                start_idx = max(0, atm_index - half_width)
                end_idx = min(len(chain_df), start_idx + max_strikes)
                # Adjust if we hit the upper bound and didn't get enough rows
                if (end_idx - start_idx) < max_strikes:
                     start_idx = max(0, end_idx - max_strikes)

                chain_df = chain_df.iloc[start_idx:end_idx].reset_index(drop=True)
                print(f"\nDisplaying {len(chain_df)} strikes around current price {self._format_currency(current_price, currency)}")

            # --- Calculate BSM, IV, Greeks ---
            results = []
            total_strikes = len(chain_df)
            for idx, row in chain_df.iterrows():
                # Progress indicator
                print(f"\rProcessing strike {idx+1}/{total_strikes}...", end="")

                strike = row['strike']
                data = {'strike': strike}

                # Call Calculations
                market_call = row.get('call_lastPrice')
                if pd.notna(market_call):
                     data['market_call'] = market_call
                     call_iv = self.calculate_implied_volatility(current_price, strike, T, risk_free_rate, market_call, "call")
                     data['call_iv'] = call_iv * 100 if pd.notna(call_iv) else np.nan
                     # Use IV for BSM if available, else use historical/input vol
                     vol_to_use_call = call_iv if pd.notna(call_iv) else volatility
                     data['bsm_call'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, vol_to_use_call, "call")
                     data['call_diff'] = market_call - data['bsm_call'] if pd.notna(data['bsm_call']) else np.nan
                     if self.config['show_greeks_in_chain']:
                          greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, vol_to_use_call, "call")
                          data['call_delta'] = greeks['delta']
                          data['call_gamma'] = greeks['gamma']
                          data['call_theta'] = greeks['theta']
                          data['call_vega'] = greeks['vega']
                else:
                    # Calculate BSM using historical/input vol if no market price
                    data['bsm_call'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "call")
                    if self.config['show_greeks_in_chain']:
                         greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, volatility, "call")
                         data['call_delta'] = greeks['delta']
                         data['call_gamma'] = greeks['gamma']
                         data['call_theta'] = greeks['theta']
                         data['call_vega'] = greeks['vega']

                # Put Calculations (similar logic)
                market_put = row.get('put_lastPrice')
                if pd.notna(market_put):
                     data['market_put'] = market_put
                     put_iv = self.calculate_implied_volatility(current_price, strike, T, risk_free_rate, market_put, "put")
                     data['put_iv'] = put_iv * 100 if pd.notna(put_iv) else np.nan
                     vol_to_use_put = put_iv if pd.notna(put_iv) else volatility
                     data['bsm_put'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, vol_to_use_put, "put")
                     data['put_diff'] = market_put - data['bsm_put'] if pd.notna(data['bsm_put']) else np.nan
                     if self.config['show_greeks_in_chain']:
                          greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, vol_to_use_put, "put")
                          data['put_delta'] = greeks['delta']
                          data['put_gamma'] = greeks['gamma']
                          data['put_theta'] = greeks['theta']
                          data['put_vega'] = greeks['vega']
                else:
                    data['bsm_put'] = self.black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "put")
                    if self.config['show_greeks_in_chain']:
                         greeks = self.calculate_option_greeks(current_price, strike, T, risk_free_rate, volatility, "put")
                         data['put_delta'] = greeks['delta']
                         data['put_gamma'] = greeks['gamma']
                         data['put_theta'] = greeks['theta']
                         data['put_vega'] = greeks['vega']


                # Add other interesting data if needed (volume, open interest)
                data['call_volume'] = row.get('call_volume', 0)
                data['call_oi'] = row.get('call_openInterest', 0)
                data['put_volume'] = row.get('put_volume', 0)
                data['put_oi'] = row.get('put_openInterest', 0)

                results.append(data)

            print("\r" + " " * 40 + "\rProcessing complete.") # Clear progress line

            results_df = pd.DataFrame(results)

             # --- Display Results ---
            pd.options.display.float_format = '{:,.2f}'.format # Nicer number formatting

            # Define columns to display
            call_cols_basic = ['call_volume', 'call_oi', 'market_call', 'bsm_call', 'call_iv']
            put_cols_basic = ['put_iv', 'bsm_put', 'market_put', 'put_oi', 'put_volume']
            strike_col = ['strike']

            call_cols_greeks = ['call_delta', 'call_gamma', 'call_theta', 'call_vega']
            put_cols_greeks = ['put_delta', 'put_gamma', 'put_theta', 'put_vega']

            display_cols = call_cols_basic + strike_col + put_cols_basic
            if self.config['show_greeks_in_chain']:
                 # Interleave greeks for better comparison
                 display_cols = ['call_volume', 'call_oi', 'market_call', 'bsm_call', 'call_iv', 'call_delta', 'call_gamma', 'call_theta', 'call_vega',
                                 'strike',
                                 'put_delta', 'put_gamma', 'put_theta', 'put_vega', 'put_iv', 'bsm_put', 'market_put', 'put_oi', 'put_volume']


            # Filter df to only include desired columns in the right order
            display_df = results_df[[col for col in display_cols if col in results_df.columns]].copy()

            # Custom formatting for the table
            formatters = {}
            currency_cols = ['market_call', 'bsm_call', 'call_diff', 'market_put', 'bsm_put', 'put_diff', 'strike', 'call_theta', 'put_theta', 'call_vega', 'put_vega']
            percent_cols = ['call_iv', 'put_iv']
            greek_cols = ['call_delta', 'call_gamma', 'put_delta', 'put_gamma']
            int_cols = ['call_volume', 'call_oi', 'put_volume', 'put_oi']

            for col in display_df.columns:
                 if col in currency_cols:
                       formatters[col] = lambda x, c=currency: self._format_currency(x, c) if pd.notna(x) else 'N/A'
                 elif col in percent_cols:
                       formatters[col] = lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A'
                 elif col in greek_cols:
                       formatters[col] = lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A'
                 elif col in int_cols:
                       formatters[col] = lambda x: f"{int(x):,}" if pd.notna(x) and x!=0 else ('0' if x==0 else 'N/A') # Format ints with comma, handle 0 and NaN

            # Rename columns for better display
            col_rename = {
                 'call_volume': 'C Vol', 'call_oi': 'C OI', 'market_call': 'C Market', 'bsm_call': 'C BSM', 'call_iv': 'C IV%',
                 'call_delta': 'C Delta', 'call_gamma': 'C Gamma', 'call_theta': 'C Theta', 'call_vega': 'C Vega',
                 'strike': 'Strike',
                 'put_delta': 'P Delta', 'put_gamma': 'P Gamma', 'put_theta': 'P Theta', 'put_vega': 'P Vega',
                 'put_iv': 'P IV%', 'bsm_put': 'P BSM', 'market_put': 'P Market', 'put_oi': 'P OI', 'put_volume': 'P Vol'
            }
            display_df.rename(columns=col_rename, inplace=True)


            print(f"\n--- Options Chain for {self.current_ticker} ---")
            print(f"Expiration: {expiration_date} ({days_to_expiration} days)")
            print(f"Current Price: {self._format_currency(current_price, currency)}")
            print(f"Risk-Free Rate: {risk_free_rate*100:.2f}%")
            print(f"Historical Vol: {volatility*100:.2f}% (used for BSM if IV unavailable)")
            print("-" * 80)

            # Use tabulate for printing
            print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f", numalign="right", stralign="right")) # , formatters=formatters doesn't work with tabulate directly

            # # Ask to visualize
            # visualize = input("\nVisualize this options chain (Price/IV)? (y/n): ").lower()
            # if visualize == 'y':
            #     self.visualize_options_chain(results_df, current_price, currency, expiration_date)

            return results_df # Return the raw calculated data

        except AttributeError:
             print(f"\nError: Could not find options data for {self.current_ticker} on {expiration_date}. The ticker might not have options or data is unavailable.")
             return None
        except Exception as e:
            print(f"\nError calculating options chain: {e}")
            if self.config['debug_mode']:
                import traceback
                traceback.print_exc()
            return None

    def visualize_options_chain(self, df, current_price, currency, expiration_date):
        """Visualize the options chain data using matplotlib."""
        if df is None or df.empty:
            print("No data available to visualize.")
            return

        df = df.copy() # Work on a copy
        df = df.dropna(subset=['strike']) # Ensure strikes are valid

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{self.current_ticker} Options Chain ({expiration_date}) - Current Price: {self._format_currency(current_price, currency)}", fontsize=16)

        # --- Plot 1: Prices (Market vs. BSM) ---
        ax1 = axes[0]
        # Calls
        if 'market_call' in df.columns:
            ax1.plot(df['strike'], df['market_call'], 'bo-', label='Market Call', markersize=5, alpha=0.7)
        if 'bsm_call' in df.columns:
             ax1.plot(df['strike'], df['bsm_call'], 'b--', label='BSM Call', alpha=0.7)
        # Puts
        if 'market_put' in df.columns:
             ax1.plot(df['strike'], df['market_put'], 'ro-', label='Market Put', markersize=5, alpha=0.7)
        if 'bsm_put' in df.columns:
             ax1.plot(df['strike'], df['bsm_put'], 'r--', label='BSM Put', alpha=0.7)

        ax1.set_ylabel(f'Option Price ({currency})')
        ax1.set_title('Market Price vs. BSM Price')
        ax1.legend()
        ax1.grid(True)
        # Add vertical line for current price
        ax1.axvline(current_price, color='grey', linestyle=':', lw=2, label=f'Current Price ({self._format_currency(current_price, currency)})')
        ax1.legend() # Show legend again to include vline label

        # --- Plot 2: Implied Volatility ---
        ax2 = axes[1]
        if 'call_iv' in df.columns and df['call_iv'].notna().any():
            ax2.plot(df['strike'], df['call_iv'], 'go-', label='Call Implied Volatility (%)', markersize=5)
        if 'put_iv' in df.columns and df['put_iv'].notna().any():
            ax2.plot(df['strike'], df['put_iv'], 'mo-', label='Put Implied Volatility (%)', markersize=5)

        # Add horizontal line for historical volatility if available
        if self.current_stock_data and self.current_stock_data['volatility'] is not None:
             hist_vol = self.current_stock_data['volatility'] * 100
             ax2.axhline(hist_vol, color='black', linestyle='--', lw=1, label=f'Hist. Vol ({hist_vol:.2f}%)')


        ax2.set_xlabel('Strike Price')
        ax2.set_ylabel('Implied Volatility (%)')
        ax2.set_title('Implied Volatility Smile / Skew')
        ax2.legend()
        ax2.grid(True)
        # Add vertical line for current price
        ax2.axvline(current_price, color='grey', linestyle=':', lw=2)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        try:
             plt.show()
        except Exception as e:
             print(f"\nError displaying plot: {e}. You might need to configure your matplotlib backend.")
             print("Try running this in an environment that supports GUI popups.")

    # --- Options Strategy Analysis ---

    def _calculate_payoff(self, S_T, strategy_legs, S0):
        """Calculates the payoff of a strategy at a given underlying price S_T."""
        total_payoff = 0
        initial_cost = 0

        for leg in strategy_legs:
            leg_type = leg['type']      # 'stock', 'call', 'put'
            direction = leg['dir']      # 'long', 'short'
            K = leg.get('K')            # Strike price (for options)
            price = leg.get('price', 0) # Cost/premium of the leg

            # Calculate initial cost/credit
            if direction == 'long':
                 initial_cost += price
            else: # short
                 initial_cost -= price # Credit received reduces cost

            # Calculate payoff at expiration S_T
            payoff_leg = 0
            if leg_type == 'stock':
                # Payoff is the change in stock price from the initial price S0
                # However, payoff diagrams usually show P/L relative to initial cost,
                # so the stock payoff is just S_T. The cost is handled by initial_cost.
                 payoff_leg = S_T
            elif leg_type == 'call':
                if direction == 'long':
                    payoff_leg = max(0, S_T - K)
                else: # short
                    payoff_leg = -max(0, S_T - K)
            elif leg_type == 'put':
                if direction == 'long':
                    payoff_leg = max(0, K - S_T)
                else: # short
                    payoff_leg = -max(0, K - S_T)

            total_payoff += payoff_leg

        # Profit/Loss = Final Payoff - Initial Cost (or + Initial Credit)
        profit_loss = total_payoff - initial_cost
        return profit_loss

    def _plot_payoff(self, S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency):
        """Plots the Profit/Loss diagram for a strategy."""
        plt.figure(figsize=(10, 6))
        plt.plot(S_T_range, PnL, lw=2, label='Profit/Loss at Expiration')

        # Mark zero profit line
        plt.axhline(0, color='black', linestyle='--', lw=1, label='Breakeven Level')

        # Mark breakeven points
        if breakevens:
            valid_bes = [be for be in breakevens if pd.notna(be)] # Filter out NaNs
            plt.scatter(valid_bes, [0] * len(valid_bes), color='red', s=100, zorder=5, label='Breakeven(s)')
            for be in valid_bes:
                 plt.text(be, 0.1 * max(abs(PnL.min()), abs(PnL.max())), f' BE: {self._format_currency(be, currency)}', color='red', ha='center')


        # Annotate Max Profit / Max Loss
        if pd.notna(max_profit) and max_profit != float('inf'):
            plt.axhline(max_profit, color='green', linestyle=':', lw=1, label=f'Max Profit: {self._format_currency(max_profit, currency)}')
        if pd.notna(max_loss) and max_loss != float('-inf'):
             plt.axhline(max_loss, color='red', linestyle=':', lw=1, label=f'Max Loss: {self._format_currency(max_loss, currency)}')


        plt.title(f'{strategy_name} Payoff Diagram')
        plt.xlabel(f'Underlying Price at Expiration ({currency})')
        plt.ylabel(f'Profit / Loss ({currency})')
        plt.grid(True)
        plt.legend()
        plt.ylim(min(PnL.min(), max_loss if pd.notna(max_loss) else PnL.min()) * 1.2,
                 max(PnL.max(), max_profit if pd.notna(max_profit) else PnL.max()) * 1.2) # Dynamic Y-axis limits

        try:
             plt.show()
        except Exception as e:
             print(f"\nError displaying plot: {e}.")


    def analyze_strategy(self):
        """Guides user through selecting and analyzing an options strategy."""
        if self.current_stock_data is None:
            print("\nPlease fetch stock data first (Option 1).")
            ticker = input("Enter stock ticker symbol: ").upper()
            if not ticker: return
            if not self.get_stock_data(ticker):
                return
        else:
             print(f"\nCurrent ticker: {self.current_ticker}")
             change_ticker = input("Analyze strategy for a different ticker? (y/n, default n): ").lower()
             if change_ticker == 'y':
                  ticker = input("Enter new stock ticker symbol: ").upper()
                  if not ticker: return
                  if not self.get_stock_data(ticker):
                       return

        stock_data = self.current_stock_data
        S0 = stock_data['current_price'] # Price at time of analysis
        expirations = stock_data['expirations']
        currency = stock_data['currency']

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
                if not (0 <= strategy_choice <= 6): # Adjust range as strategies are added
                    print("Invalid choice.")
                    strategy_choice = None
            except ValueError:
                print("Invalid input. Please enter a number.")

        if strategy_choice == 0:
             return

        # --- Select Expiration ---
        expiration_date = self._select_expiration_date(expirations)
        if not expiration_date:
            return

        # --- Get Strategy Specific Parameters ---
        strategy_legs = []
        strategy_name = ""
        breakevens = []
        max_profit = np.nan
        max_loss = np.nan

        try:
            # --- Covered Call ---
            if strategy_choice == 1:
                strategy_name = "Covered Call"
                print(f"\n--- {strategy_name} Setup ---")
                print(f"Action: Buy 100 shares of {self.current_ticker} and Sell 1 Call Option.")
                # Select Call Strike
                strike_input = input(f"Enter Call Strike Price (e.g., slightly OTM like {S0*1.05:.2f}): ")
                K_call = float(strike_input)
                if K_call <= 0: raise ValueError("Strike must be positive.")

                # Get Call Option Data
                call_data = self._get_option_data_for_strike(expiration_date, K_call, 'call')
                if call_data is None: raise ValueError("Could not retrieve call option data.")
                call_premium = call_data['lastPrice'] # Premium received for selling
                if pd.isna(call_premium) or call_premium <= 0:
                     print(f"Warning: Market premium for Call K={K_call} is missing or zero. Using BSM estimate.")
                     # Fallback to BSM if market price unavailable
                     vol = self.current_stock_data['volatility'] or 0.2 # Use historical or default vol
                     r = self.risk_free_rate or self.config['default_risk_free_rate']
                     today = dt.datetime.now().date()
                     exp_d = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
                     T = max(0, (exp_d - today).days) / 365.0
                     call_premium = self.black_scholes_merton(S0, K_call, T, r, vol, 'call')
                     if pd.isna(call_premium): raise ValueError("Could not estimate call premium.")


                print(f"Received premium for selling Call K={K_call}: {self._format_currency(call_premium, currency)}")

                strategy_legs = [
                    {'type': 'stock', 'dir': 'long', 'price': S0}, # Cost of stock
                    {'type': 'call', 'dir': 'short', 'K': K_call, 'price': call_premium} # Credit from call
                ]
                # Cost basis = Stock Price - Call Premium
                cost_basis = S0 - call_premium
                breakeven = cost_basis
                max_profit = (K_call - S0) + call_premium # Profit if assigned at K_call
                max_loss = -cost_basis # Loss if stock goes to 0

                print(f"Net Cost Basis: {self._format_currency(cost_basis, currency)}")
                breakevens = [breakeven]


            # --- Protective Put ---
            elif strategy_choice == 2:
                strategy_name = "Protective Put"
                print(f"\n--- {strategy_name} Setup ---")
                print(f"Action: Buy 100 shares of {self.current_ticker} and Buy 1 Put Option.")
                # Select Put Strike
                strike_input = input(f"Enter Put Strike Price (e.g., slightly OTM like {S0*0.95:.2f}): ")
                K_put = float(strike_input)
                if K_put <= 0: raise ValueError("Strike must be positive.")

                # Get Put Option Data
                put_data = self._get_option_data_for_strike(expiration_date, K_put, 'put')
                if put_data is None: raise ValueError("Could not retrieve put option data.")
                put_premium = put_data['lastPrice'] # Cost of buying put
                if pd.isna(put_premium) or put_premium <= 0:
                     print(f"Warning: Market premium for Put K={K_put} is missing or zero. Using BSM estimate.")
                     vol = self.current_stock_data['volatility'] or 0.2
                     r = self.risk_free_rate or self.config['default_risk_free_rate']
                     today = dt.datetime.now().date()
                     exp_d = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
                     T = max(0, (exp_d - today).days) / 365.0
                     put_premium = self.black_scholes_merton(S0, K_put, T, r, vol, 'put')
                     if pd.isna(put_premium): raise ValueError("Could not estimate put premium.")

                print(f"Cost for buying Put K={K_put}: {self._format_currency(put_premium, currency)}")

                strategy_legs = [
                    {'type': 'stock', 'dir': 'long', 'price': S0}, # Cost of stock
                    {'type': 'put', 'dir': 'long', 'K': K_put, 'price': put_premium} # Cost of put
                ]
                # Total cost = Stock Price + Put Premium
                total_cost = S0 + put_premium
                breakeven = total_cost
                max_profit = float('inf') # Unlimited potential profit
                max_loss = -(S0 - K_put + put_premium) # Loss if stock below K_put at expiration (S0 - BE)

                print(f"Total Cost: {self._format_currency(total_cost, currency)}")
                breakevens = [breakeven]


            # --- Bull Call Spread ---
            elif strategy_choice == 3:
                 strategy_name = "Bull Call Spread"
                 print(f"\n--- {strategy_name} Setup ---")
                 print(f"Action: Buy 1 Lower Strike Call and Sell 1 Higher Strike Call.")
                 K_low_str = input(f"Enter Lower Call Strike (Long, e.g., {S0*0.98:.2f}): ")
                 K_high_str = input(f"Enter Higher Call Strike (Short, e.g., {S0*1.02:.2f}): ")
                 K_low = float(K_low_str)
                 K_high = float(K_high_str)
                 if not (0 < K_low < K_high): raise ValueError("Strikes must be positive and Low < High.")

                 call_low_data = self._get_option_data_for_strike(expiration_date, K_low, 'call')
                 call_high_data = self._get_option_data_for_strike(expiration_date, K_high, 'call')
                 if call_low_data is None or call_high_data is None: raise ValueError("Could not get option data.")

                 prem_low = call_low_data['lastPrice'] # Cost
                 prem_high = call_high_data['lastPrice'] # Credit

                 if pd.isna(prem_low) or pd.isna(prem_high):
                      raise ValueError("Missing market price for one or both options. Cannot analyze.")

                 net_debit = prem_low - prem_high
                 print(f"Cost of Long Call K={K_low}: {self._format_currency(prem_low, currency)}")
                 print(f"Credit from Short Call K={K_high}: {self._format_currency(prem_high, currency)}")
                 print(f"Net Debit (Cost): {self._format_currency(net_debit, currency)}")

                 strategy_legs = [
                      {'type': 'call', 'dir': 'long', 'K': K_low, 'price': prem_low},
                      {'type': 'call', 'dir': 'short', 'K': K_high, 'price': prem_high}
                 ]
                 max_profit = (K_high - K_low) - net_debit
                 max_loss = -net_debit
                 breakeven = K_low + net_debit
                 breakevens = [breakeven]


            # --- Bear Put Spread ---
            elif strategy_choice == 4:
                 strategy_name = "Bear Put Spread"
                 print(f"\n--- {strategy_name} Setup ---")
                 print(f"Action: Buy 1 Higher Strike Put and Sell 1 Lower Strike Put.")
                 K_high_str = input(f"Enter Higher Put Strike (Long, e.g., {S0*1.02:.2f}): ")
                 K_low_str = input(f"Enter Lower Put Strike (Short, e.g., {S0*0.98:.2f}): ")
                 K_high = float(K_high_str)
                 K_low = float(K_low_str)
                 if not (0 < K_low < K_high): raise ValueError("Strikes must be positive and Low < High.")

                 put_high_data = self._get_option_data_for_strike(expiration_date, K_high, 'put')
                 put_low_data = self._get_option_data_for_strike(expiration_date, K_low, 'put')
                 if put_high_data is None or put_low_data is None: raise ValueError("Could not get option data.")

                 prem_high = put_high_data['lastPrice'] # Cost
                 prem_low = put_low_data['lastPrice'] # Credit

                 if pd.isna(prem_high) or pd.isna(prem_low):
                      raise ValueError("Missing market price for one or both options. Cannot analyze.")

                 net_debit = prem_high - prem_low
                 print(f"Cost of Long Put K={K_high}: {self._format_currency(prem_high, currency)}")
                 print(f"Credit from Short Put K={K_low}: {self._format_currency(prem_low, currency)}")
                 print(f"Net Debit (Cost): {self._format_currency(net_debit, currency)}")

                 strategy_legs = [
                      {'type': 'put', 'dir': 'long', 'K': K_high, 'price': prem_high},
                      {'type': 'put', 'dir': 'short', 'K': K_low, 'price': prem_low}
                 ]
                 max_profit = (K_high - K_low) - net_debit
                 max_loss = -net_debit
                 breakeven = K_high - net_debit
                 breakevens = [breakeven]

            # --- Long Straddle ---
            elif strategy_choice == 5:
                strategy_name = "Long Straddle"
                print(f"\n--- {strategy_name} Setup ---")
                print(f"Action: Buy 1 ATM Call and Buy 1 ATM Put (same strike & expiration).")
                # Find ATM strike
                try:
                     options = self.current_stock_data['ticker_object'].option_chain(expiration_date)
                     all_strikes = sorted(list(set(options.calls['strike'].tolist() + options.puts['strike'].tolist())))
                     if not all_strikes: raise ValueError("No strikes found.")
                     K_atm = min(all_strikes, key=lambda x: abs(x - S0))
                     print(f"Using closest ATM strike: {self._format_currency(K_atm, currency)}")
                except Exception as e:
                     raise ValueError(f"Could not determine ATM strike: {e}")

                call_data = self._get_option_data_for_strike(expiration_date, K_atm, 'call')
                put_data = self._get_option_data_for_strike(expiration_date, K_atm, 'put')
                if call_data is None or put_data is None: raise ValueError("Could not get ATM option data.")

                prem_call = call_data['lastPrice']
                prem_put = put_data['lastPrice']
                if pd.isna(prem_call) or pd.isna(prem_put):
                     raise ValueError("Missing market price for ATM call or put.")

                total_cost = prem_call + prem_put
                print(f"Cost of Long Call K={K_atm}: {self._format_currency(prem_call, currency)}")
                print(f"Cost of Long Put K={K_atm}: {self._format_currency(prem_put, currency)}")
                print(f"Total Cost (Net Debit): {self._format_currency(total_cost, currency)}")

                strategy_legs = [
                      {'type': 'call', 'dir': 'long', 'K': K_atm, 'price': prem_call},
                      {'type': 'put', 'dir': 'long', 'K': K_atm, 'price': prem_put}
                 ]
                max_profit = float('inf') # Unlimited potential profit
                max_loss = -total_cost # Loss is the total premium paid
                breakeven_up = K_atm + total_cost
                breakeven_down = K_atm - total_cost
                breakevens = [breakeven_down, breakeven_up]

            # --- Long Strangle ---
            elif strategy_choice == 6:
                strategy_name = "Long Strangle"
                print(f"\n--- {strategy_name} Setup ---")
                print(f"Action: Buy 1 OTM Call and Buy 1 OTM Put (different strikes, same expiration).")
                K_call_str = input(f"Enter OTM Call Strike (Long, e.g., {S0*1.05:.2f}): ")
                K_put_str = input(f"Enter OTM Put Strike (Long, e.g., {S0*0.95:.2f}): ")
                K_call = float(K_call_str)
                K_put = float(K_put_str)
                if not (0 < K_put < S0 < K_call): # Basic check for OTM
                     print("Warning: Ensure Call Strike > Current Price > Put Strike for typical strangle.")
                if K_put <= 0 or K_call <= 0: raise ValueError("Strikes must be positive.")

                call_data = self._get_option_data_for_strike(expiration_date, K_call, 'call')
                put_data = self._get_option_data_for_strike(expiration_date, K_put, 'put')
                if call_data is None or put_data is None: raise ValueError("Could not get option data.")

                prem_call = call_data['lastPrice']
                prem_put = put_data['lastPrice']
                if pd.isna(prem_call) or pd.isna(prem_put):
                     raise ValueError("Missing market price for one or both options.")

                total_cost = prem_call + prem_put
                print(f"Cost of Long Call K={K_call}: {self._format_currency(prem_call, currency)}")
                print(f"Cost of Long Put K={K_put}: {self._format_currency(prem_put, currency)}")
                print(f"Total Cost (Net Debit): {self._format_currency(total_cost, currency)}")

                strategy_legs = [
                      {'type': 'call', 'dir': 'long', 'K': K_call, 'price': prem_call},
                      {'type': 'put', 'dir': 'long', 'K': K_put, 'price': prem_put}
                 ]
                max_profit = float('inf')
                max_loss = -total_cost
                breakeven_up = K_call + total_cost
                breakeven_down = K_put - total_cost
                breakevens = [breakeven_down, breakeven_up]


            # --- Common Calculation & Plotting ---
            if strategy_legs:
                # Define range for underlying price at expiration
                price_range = self.config['strategy_price_range'] # e.g., 0.3 for +/- 30%
                S_T_min = S0 * (1 - price_range)
                S_T_max = S0 * (1 + price_range)
                # Ensure range covers breakevens if they are outside the default range
                if breakevens:
                     S_T_min = min(S_T_min, min(b for b in breakevens if pd.notna(b)) * 0.9)
                     S_T_max = max(S_T_max, max(b for b in breakevens if pd.notna(b)) * 1.1)

                S_T_range = np.linspace(max(0, S_T_min), S_T_max, 100) # Ensure price >= 0

                # Calculate P/L for each price in the range
                PnL = np.array([self._calculate_payoff(s_t, strategy_legs, S0) for s_t in S_T_range])

                # Display Summary
                print("\n--- Strategy Summary ---")
                print(f"Strategy: {strategy_name}")
                print(f"Expiration: {expiration_date}")
                print(f"Current Price: {self._format_currency(S0, currency)}")
                for i, leg in enumerate(strategy_legs):
                     k_str = f" K={self._format_currency(leg['K'], currency)}" if 'K' in leg else ""
                     p_str = f" @ {self._format_currency(leg['price'], currency)}"
                     print(f" Leg {i+1}: {leg['dir'].capitalize()} {leg['type'].capitalize()}{k_str}{p_str}")

                be_str = ", ".join([self._format_currency(be, currency) for be in breakevens if pd.notna(be)]) or "N/A"
                mp_str = self._format_currency(max_profit, currency) if pd.notna(max_profit) and max_profit != float('inf') else ('Unlimited' if max_profit == float('inf') else 'N/A')
                ml_str = self._format_currency(max_loss, currency) if pd.notna(max_loss) and max_loss != float('-inf') else ('Unlimited' if max_loss == float('-inf') else 'N/A')

                print(f"\nBreakeven(s): {be_str}")
                print(f"Max Profit: {mp_str}")
                print(f"Max Loss: {ml_str}")

                # Plot Payoff Diagram
                self._plot_payoff(S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency)

        except ValueError as ve:
             print(f"\nInput Error: {ve}")
        except Exception as e:
             print(f"\nAn error occurred during strategy analysis: {e}")
             if self.config['debug_mode']:
                  import traceback
                  traceback.print_exc()


    # --- Menu and Application Flow ---

    def manage_favorites(self):
        """Manage the list of favorite tickers."""
        while True:
            self.clear_screen()
            print("--- Manage Favorite Tickers ---")
            if not self.favorite_tickers:
                print("No favorite tickers saved.")
            else:
                print("Current Favorites:")
                for i, ticker in enumerate(self.favorite_tickers):
                    print(f" {i+1}. {ticker}")

            print("\nOptions:")
            print(" 1. Add Ticker")
            print(" 2. Remove Ticker")
            print(" 0. Back to Main Menu")

            choice = input("Enter option: ")

            if choice == '1':
                ticker_to_add = input("Enter ticker symbol to add: ").upper().strip()
                if ticker_to_add and self.validate_ticker(ticker_to_add):
                    if ticker_to_add not in self.favorite_tickers:
                        self.favorite_tickers.append(ticker_to_add)
                        self.favorite_tickers.sort() # Keep list sorted
                        self._save_favorite_tickers()
                        print(f"'{ticker_to_add}' added to favorites.")
                    else:
                        print(f"'{ticker_to_add}' is already in favorites.")
                elif ticker_to_add:
                     print(f"Could not validate '{ticker_to_add}'. Not added.")
                input("Press Enter to continue...")
            elif choice == '2':
                if not self.favorite_tickers:
                    print("No favorites to remove.")
                    input("Press Enter to continue...")
                    continue
                try:
                    num_to_remove = int(input("Enter the number of the ticker to remove: "))
                    if 1 <= num_to_remove <= len(self.favorite_tickers):
                        removed_ticker = self.favorite_tickers.pop(num_to_remove - 1)
                        self._save_favorite_tickers()
                        print(f"'{removed_ticker}' removed from favorites.")
                    else:
                        print("Invalid number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                input("Press Enter to continue...")
            elif choice == '0':
                break
            else:
                print("Invalid option.")
                input("Press Enter to continue...")

    def manage_settings(self):
         """Allow user to view and modify configuration settings."""
         while True:
             self.clear_screen()
             print("--- Configure Settings ---")
             settings_list = list(self.config.items())

             for i, (key, value) in enumerate(settings_list):
                  print(f" {i+1}. {key}: {value}")

             print("\n 0. Back to Main Menu (Save Changes)")

             choice = input("\nEnter the number of the setting to change (or 0 to save and exit): ")

             try:
                  choice_idx = int(choice)
                  if choice_idx == 0:
                       self._save_config()
                       break
                  elif 1 <= choice_idx <= len(settings_list):
                       setting_key, current_value = settings_list[choice_idx - 1]
                       new_value_str = input(f"Enter new value for '{setting_key}' (current: {current_value}): ")

                       # Try to convert to the correct type
                       try:
                            if isinstance(current_value, bool):
                                 if new_value_str.lower() in ['true', 't', 'yes', 'y', '1']:
                                      new_value = True
                                 elif new_value_str.lower() in ['false', 'f', 'no', 'n', '0']:
                                      new_value = False
                                 else:
                                      raise ValueError("Invalid boolean value")
                            elif isinstance(current_value, int):
                                 new_value = int(new_value_str)
                            elif isinstance(current_value, float):
                                 new_value = float(new_value_str)
                            else: # Assume string
                                  new_value = new_value_str

                            self.config[setting_key] = new_value
                            print(f"Set '{setting_key}' to '{new_value}'.")

                       except ValueError:
                            print(f"Invalid value type entered for '{setting_key}'. Expected type: {type(current_value).__name__}. No change made.")

                       input("Press Enter to continue...")
                  else:
                       print("Invalid selection.")
                       input("Press Enter to continue...")

             except ValueError:
                  print("Invalid input. Please enter a number.")
                  input("Press Enter to continue...")


    def display_main_menu(self):
        """Display the main menu options."""
        self.clear_screen()
        print("===== Options Analyzer Menu =====")
        if self.current_ticker:
            print(f"Current Ticker: {self.current_ticker} ({self._format_currency(self.current_stock_data['current_price'], self.current_stock_data['currency'])})")
        else:
            print("Current Ticker: None")
        print("-" * 30)
        print("1. Fetch Stock Data / Change Ticker")
        print("2. Simple Option Price (BSM & Greeks)")
        print("3. View Options Chain (Market vs. BSM, IV)")
        print("4. Analyze Option Strategy (Payoff Diagrams)")
        print("5. Manage Favorite Tickers")
        print("6. Configure Settings")
        print("0. Exit")
        print("-" * 30)
        if self.favorite_tickers:
             fav_str = ", ".join(self.favorite_tickers[:5]) # Show first 5 favorites
             if len(self.favorite_tickers) > 5: fav_str += "..."
             print(f"Favorites: {fav_str}")


    def run(self):
        """Main application loop."""
        while True:
            self.display_main_menu()
            choice = input("Enter your choice: ")

            if choice == '1':
                # Ask for ticker, suggest current or favorites
                prompt = "Enter ticker symbol"
                if self.current_ticker:
                     prompt += f" (Enter for '{self.current_ticker}'"
                     if self.favorite_tickers:
                          prompt += f", or type fav)"
                     else:
                          prompt += ")"
                elif self.favorite_tickers:
                     prompt += " (or type fav)"

                ticker_input = input(f"{prompt}: ").upper().strip()

                selected_ticker = None
                if not ticker_input and self.current_ticker:
                     selected_ticker = self.current_ticker # Keep current
                     print(f"Keeping current ticker: {selected_ticker}")
                     # Re-fetch data in case it's stale? Optional. For now, just keep.
                     # self.get_stock_data(selected_ticker)
                elif ticker_input == 'FAV' and self.favorite_tickers:
                     print("\nFavorites:")
                     for i, fav in enumerate(self.favorite_tickers):
                          print(f" {i+1}. {fav}")
                     fav_choice = input("Select favorite number: ")
                     try:
                          idx = int(fav_choice) - 1
                          if 0 <= idx < len(self.favorite_tickers):
                               selected_ticker = self.favorite_tickers[idx]
                          else:
                               print("Invalid favorite number.")
                     except ValueError:
                          print("Invalid input.")
                elif ticker_input:
                      selected_ticker = ticker_input

                if selected_ticker:
                    self.get_stock_data(selected_ticker) # Fetch/update data
                elif not ticker_input and not self.current_ticker:
                     print("No ticker entered.")


            elif choice == '2':
                self.get_simple_option_price()
            elif choice == '3':
                self.calculate_options_chain()
            elif choice == '4':
                 self.analyze_strategy()
            elif choice == '5':
                 self.manage_favorites()
            elif choice == '6':
                 self.manage_settings()
            elif choice == '0':
                print("Exiting Options Analyzer. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

            input("\nPress Enter to return to the Main Menu...")


if __name__ == "__main__":
    analyzer = OptionsAnalyzer()
    analyzer.run()