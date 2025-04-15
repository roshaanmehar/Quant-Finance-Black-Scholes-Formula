
# standard imports 
import numpy as np
import pandas as pd #for data manipulation and analysis
import yfinance as yf #for fetching financial data from yahoo finance api
import datetime as dt 
from scipy.stats import norm #for statistical calculations and functions
from tabulate import tabulate #to tabulate data in the console


class OptionsAnalyzer:
    def __init__(self):
        """Initialize the Options Analyzer with default parameters"""
        self.current_ticker = None
        self.current_stock_data = None
        self.risk_free_rate = None
        self.config = self._load_config()
        self.favorite_tickers = self._load_favorite_tickers()
        # Fetch initial risk-free rate
        self.get_risk_free_rate()
    
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
def validate_ticker(ticker):
    """Validate if the ticker exists"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return False
        return True
    except:
        return False




def visualize_options_chain(df, current_price):
    """
    Visualize the options chain using matplotlib
    """
    import matplotlib.pyplot as plt
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot call prices
    ax1.plot(df['Strike'], df['BSM Call'], 'b-', label='BSM Model')
    if 'Market Call' in df.columns and not df['Market Call'].isna().all():
        ax1.scatter(df['Strike'], df['Market Call'], color='r', label='Market')
    ax1.axvline(x=current_price, color='g', linestyle='--', label='Current Price')
    ax1.set_title('Call Option Prices')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Option Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot put prices
    ax2.plot(df['Strike'], df['BSM Put'], 'b-', label='BSM Model')
    if 'Market Put' in df.columns and not df['Market Put'].isna().all():
        ax2.scatter(df['Strike'], df['Market Put'], color='r', label='Market')
    ax2.axvline(x=current_price, color='g', linestyle='--', label='Current Price')
    ax2.set_title('Put Option Prices')
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Option Price')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n===== Options Analysis Tool =====")
        print("1. Calculate full options chain")
        print("2. Get simple option price")
        print("3. Calculate option Greeks")
        print("4. Calculate implied volatility")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper()
            calculate_options_chain(ticker)
        elif choice == "2":
            ticker = input("Enter stock ticker symbol: ").upper()
            option_type = input("Option type (call/put/both): ").lower() or "both"
            strike_input = input("Strike price (enter 'atm' for at-the-money or a specific price): ").lower() or "atm"
            get_simple_option_price(ticker, option_type, strike_input)
        elif choice == "3":
            # Add code for calculating Greeks for a specific option
            pass
        elif choice == "4":
            # Add code for calculating implied volatility
            pass
        elif choice == "5":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

def calculate_implied_volatility(S, K, T, r, market_price, option_type="call", precision=0.0001):
    """
    Calculate implied volatility using bisection method
    """
    max_iterations = 100
    vol_low = 0.001
    vol_high = 5.0  # 500% volatility as upper bound
    
    for i in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2
        price = black_scholes_merton(S, K, T, r, vol_mid, option_type)
        
        if abs(price - market_price) < precision:
            return vol_mid
        
        if price > market_price:
            vol_high = vol_mid
        else:
            vol_low = vol_mid
    
    return (vol_low + vol_high) / 2


def calculate_option_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option Greeks using Black-Scholes-Merton model
    
    Returns:
    Dictionary with Delta, Gamma, Theta, Vega, Rho
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta - sensitivity to underlying price change
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma - rate of change of Delta
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta - sensitivity to time decay (per day)
    part1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type.lower() == "call":
        part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (part1 + part2) / 365  # Convert to daily
    else:
        part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (part1 + part2) / 365  # Convert to daily
    
    # Vega - sensitivity to volatility change (for 1% change)
    vega = S * np.sqrt(T) * norm.pdf(d1) / 100
    
    # Rho - sensitivity to interest rate change (for 1% change)
    if option_type.lower() == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

def get_simple_option_price(ticker, option_type="both", strike_type="atm"):
    """
    Get a simple option price for a specified ticker
    
    Parameters:
    ticker: Stock ticker symbol
    option_type: "call", "put", or "both"
    strike_type: "atm" (at-the-money), or a specific price
    
    Returns:
    Option price(s)
    """
    stock_data = get_stock_data(ticker)
    if not stock_data:
        return
    
    current_price = stock_data['current_price']
    volatility = stock_data['volatility']
    expirations = stock_data['expirations']
    stock = stock_data['ticker_object']
    
    # Get risk-free rate
    risk_free_rate = get_risk_free_rate()
    
    if not expirations:
        print("No options data available for this ticker.")
        return
    
    # Use the nearest expiration date
    expiration_date = expirations[0]
    print(f"Using expiration date: {expiration_date}")
    
    # Calculate time to expiration in years
    today = dt.datetime.now().date()
    exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
    days_to_expiration = (exp_date - today).days
    T = days_to_expiration / 365
    
    # Determine strike price
    if strike_type == "atm":
        strike = round(current_price / 5) * 5  # Round to nearest $5 increment
    else:
        try:
            strike = float(strike_type)
        except:
            strike = current_price  # Default to current price if invalid input
    
    # Calculate option prices
    call_price = black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "call")
    put_price = black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "put")
    
    if option_type.lower() == "call":
        print(f"Call option price for {ticker} at strike ${strike:.2f}: ${call_price:.2f}")
        return call_price
    elif option_type.lower() == "put":
        print(f"Put option price for {ticker} at strike ${strike:.2f}: ${put_price:.2f}")
        return put_price
    else:
        print(f"Option prices for {ticker} at strike ${strike:.2f}:")
        print(f"Call: ${call_price:.2f}")
        print(f"Put: ${put_price:.2f}")
        return {"call": call_price, "put": put_price}

def get_stock_data(ticker):
    """Fetch stock data and options chain using Yahoo Finance API"""
    try:
        stock = yf.Ticker(ticker) #initializes a ticker object for the given stock
        hist = stock.history(period="1y") #fetches stock history for the given period
        
        if hist.empty:
            raise ValueError(f"Could not fetch data for {ticker}") #raises an error if no history is found
        
        current_price = hist['Close'].iloc[-1] #calculates the stock price by extracting the closing price of the most recent trading day
        
        # Calculate historical volatility (annualized)
        returns = np.log(hist['Close'] / hist['Close'].shift(1)) #calculates the daily logarithmic returns of the stock
        volatility = returns.std() * np.sqrt(252)  # computes the annualized standard deviation of the returns to estimate volatility
        
        print(f"Current price for {ticker}: ${current_price:.2f}")
        print(f"Historical volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        
        # Get available expiration dates
        expirations = stock.options #retrieves a list of available option expiration dates for the stock
        
        return {
            'current_price': current_price,
            'volatility': volatility,
            'expirations': expirations,
            'ticker_object': stock
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_risk_free_rate():
    """Get risk-free rate from Treasury yield (10-year as proxy)"""
    try:
        # attempts tp retreive the current risk free interest rate using the 10 year U.S treasurey yeild as a proxy
        treasury = yf.Ticker("^TNX") #initializes a ticker object for the 10-year U.S. Treasury yield
        data = treasury.history(period="1d") 
        if not data.empty:
            # Convert from percentage to decimal
            rate = data['Close'].iloc[-1] / 100
            print(f"Current risk-free rate: {rate:.4f} ({rate*100:.2f}%)")
            return rate
    except:
        pass
    
    # Fallback to a reasonable default if API fails
    default_rate = 0.04  # 4%
    print(f"Using default risk-free rate: {default_rate:.4f} ({default_rate*100:.2f}%)")
    return default_rate

def black_scholes_merton(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option price using Black-Scholes-Merton model
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Volatility of the stock
    option_type: "call" or "put"
    
    Returns:
    Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_options_chain(ticker):
    """Calculate a full options chain for a given stock"""
    # Get stock data
    stock_data = get_stock_data(ticker)
    if not stock_data:
        return
    
    current_price = stock_data['current_price']
    volatility = stock_data['volatility']
    expirations = stock_data['expirations']
    stock = stock_data['ticker_object']
    
    # Get risk-free rate
    risk_free_rate = get_risk_free_rate()
    
    if not expirations:
        print("No options data available for this ticker.")
        return
    
    # Print available expiration dates
    print("\nAvailable expiration dates:")
    for i, date in enumerate(expirations):
        print(f"{i+1}. {date}")
    
    # Let user select an expiration date
    try:
        selection = int(input("\nSelect expiration date (number): ")) - 1
        if selection < 0 or selection >= len(expirations):
            raise ValueError("Invalid selection")
        expiration_date = expirations[selection]
    except (ValueError, IndexError):
        print("Invalid selection. Using first available date.")
        expiration_date = expirations[0]
    
    print(f"\nSelected expiration date: {expiration_date}")
    
    # Calculate time to expiration in years
    today = dt.datetime.now().date()
    exp_date = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
    days_to_expiration = (exp_date - today).days
    T = days_to_expiration / 365
    
    print(f"Days to expiration: {days_to_expiration}")
    
    # Get option chain from Yahoo Finance
    try:
        options = stock.option_chain(expiration_date)
        calls = options.calls
        puts = options.puts
        
        # Get unique strike prices
        strikes = sorted(set(calls['strike'].tolist()))
        
        # Create a DataFrame to store our calculated values
        results = []
        
        print("\nCalculating option prices using Black-Scholes-Merton model...\n")
        
        for strike in strikes:
            # Calculate theoretical prices
            bsm_call = black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "call")
            bsm_put = black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "put")
            
            # Get market prices if available
            market_call = calls[calls['strike'] == strike]['lastPrice'].values[0] if not calls[calls['strike'] == strike].empty else None
            market_put = puts[puts['strike'] == strike]['lastPrice'].values[0] if not puts[puts['strike'] == strike].empty else None
            
            # Calculate difference between model and market
            call_diff = market_call - bsm_call if market_call is not None else None
            put_diff = market_put - bsm_put if market_put is not None else None
            
            results.append({
                'Strike': strike,
                'BSM Call': bsm_call,
                'Market Call': market_call,
                'Call Diff': call_diff,
                'BSM Put': bsm_put,
                'Market Put': market_put,
                'Put Diff': put_diff
            })
        
        # Convert to DataFrame and display
        df = pd.DataFrame(results)
        
        # Format the output
        pd.set_option('display.float_format', '${:.2f}'.format)
        
        # Display the results
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        
        # Find the at-the-money option
        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
        atm_row = df[df['Strike'] == atm_strike]
        
        print(f"\nAt-the-money (closest to current price ${current_price:.2f}):")
        print(tabulate(atm_row, headers='keys', tablefmt='psql', showindex=False))
        
        return df
        
    except Exception as e:
        print(f"Error calculating options chain: {e}")
        
        # Fallback to manual calculation if Yahoo Finance options data fails
        print("\nFalling back to manual calculation...")
        
        # Generate strikes around current price
        strikes = np.linspace(current_price * 0.8, current_price * 1.2, 9)
        
        results = []
        for strike in strikes:
            bsm_call = black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "call")
            bsm_put = black_scholes_merton(current_price, strike, T, risk_free_rate, volatility, "put")
            
            results.append({
                'Strike': strike,
                'Call Price': bsm_call,
                'Put Price': bsm_put
            })
        
        df = pd.DataFrame(results)
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        
        return df

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper()
    calculate_options_chain(ticker)