
# standard imports 
import numpy as np
import pandas as pd #for data manipulation and analysis
import yfinance as yf #for fetching financial data from yahoo finance api
import datetime as dt 
from scipy.stats import norm #for statistical calculations and functions
from tabulate import tabulate #to tabulate data in the console

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