# data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import traceback
from utils import safe_float, format_currency

def validate_ticker(ticker, debug_mode=False):
    """Validate if the ticker exists using yfinance."""
    if not ticker or not isinstance(ticker, str):
        print("Ticker symbol must be a non-empty string.")
        return False, None
    ticker = ticker.upper().strip()
    try:
        print(f"\nValidating ticker '{ticker}'...")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d") # Check recent history first
        info = stock.info # Fetch info regardless

        if hist.empty and (not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
             print(f"Ticker '{ticker}' invalid/no recent data.")
             return False, None

        q_type = info.get('quoteType', 'N/A')
        allowed = ['EQUITY', 'ETF', 'INDEX', 'CURRENCY', 'COMMODITY']
        if q_type not in allowed: print(f"Warn: Type '{q_type}'. Options may differ.")

        print(f"Ticker '{ticker}' valid ({info.get('shortName', 'N/A')}).")
        return True, stock # Return the ticker object as well if valid

    except Exception as e:
        print(f"Error validating ticker '{ticker}': {e}")
        if debug_mode: traceback.print_exc()
        return False, None

def get_stock_data(ticker, stock_object=None, debug_mode=False):
    """Fetch stock data, info, yield, expirations. Can reuse stock_object."""
    if not stock_object:
        is_valid, stock_object = validate_ticker(ticker, debug_mode)
        if not is_valid: return None
    elif not isinstance(stock_object, yf.Ticker):
         print("Error: Invalid stock_object passed.")
         return None

    ticker = stock_object.ticker # Get official ticker from object

    try:
        print(f"\nFetching data for {ticker}...")
        hist = stock_object.history(period="1y", interval="1d")
        price, vol = None, None # Initialize as None

        if not hist.empty and 'Close' in hist.columns:
            hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
            hist.dropna(subset=['Close'], inplace=True)
            if not hist.empty:
                price = safe_float(hist['Close'].iloc[-1])
                returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                if len(returns) >= 10: vol = safe_float(returns.std() * np.sqrt(252)) # Use fixed 252 days
                else: print(f"Warn: Only {len(returns)} returns for vol calc.")
        else: print(f"Warn: History issue for {ticker}.")

        info = {}; name=ticker; sector, industry, mcap_str, currency = 'N/A', 'N/A', 'N/A', 'USD'; div_yield = 0.0
        try:
            info = stock_object.info; name = info.get('shortName', ticker); sector = info.get('sector', 'N/A'); industry = info.get('industry', 'N/A')
            mcap = safe_float(info.get('marketCap')); currency = info.get('currency', 'USD'); div_yield = safe_float(info.get('dividendYield', 0.0), 0.0)
            if price is None or np.isnan(price): # Fallback price logic
                 for k in ['regularMarketPrice', 'currentPrice', 'previousClose', 'regularMarketOpen']:
                     p_val = safe_float(info.get(k));
                     if p_val is not None and not np.isnan(p_val) and p_val > 0: price = p_val; print(f"Using price from '{k}'"); break
                 if price is None or np.isnan(price):
                     hi,lo = safe_float(info.get('dayHigh')), safe_float(info.get('dayLow'));
                     if pd.notna(hi) and pd.notna(lo) and hi >= lo > 0: price = (hi+lo)/2; print("Warn: Using day hi/lo midpoint.")
            if mcap is not None and not np.isnan(mcap): # Format market cap
                if mcap >= 1e12: mcap_str = f"{format_currency(mcap / 1e12, currency)}T"
                elif mcap >= 1e9: mcap_str = f"{format_currency(mcap / 1e9, currency)}B"
                elif mcap >= 1e6: mcap_str = f"{format_currency(mcap / 1e6, currency)}M"
                else: mcap_str = format_currency(mcap, currency)
        except Exception as e_info: print(f"Warn: Info fetch issue: {e_info}");

        if price is None or np.isnan(price): raise ValueError("CRITICAL: Failed to find price.")

        print(f"\n=== {name} ({ticker}) ==="); print(f"Price: {format_currency(price, currency)}")
        print(f"Sector: {sector} | Industry: {industry}"); print(f"Market Cap: {mcap_str}")
        print(f"Dividend Yield: {div_yield:.4f} ({div_yield*100:.2f}%)")
        if vol is not None and not np.isnan(vol): print(f"Volatility (1y): {vol:.4f} ({vol*100:.2f}%)")
        else: print("Volatility (1y): N/A")

        expirations = ()
        try: expirations = stock_object.options; print(f"Found {len(expirations)} expirations.") if expirations else print("Note: No options found.")
        except Exception as e: print(f"Warn: Opt fetch issue: {e}")

        print(f"Data fetch complete.");
        return {'ticker': ticker, 'current_price': price, 'volatility': vol, 'dividend_yield': div_yield,
                'expirations': expirations, 'ticker_object': stock_object, 'history': hist, 'info': info, 'currency': currency}

    except Exception as e:
        print(f"\nError getting data for '{ticker}': {e}");
        if debug_mode: traceback.print_exc(); return None

def get_risk_free_rate(default_rate=0.04, verbose=False, debug_mode=False):
    """Fetches 10Y Treasury yield (^TNX) or returns default."""
    try:
        if verbose: print("Fetching risk-free rate (^TNX)...")
        tnx = yf.Ticker("^TNX"); data = tnx.download(period="5d", progress=False)
        if not data.empty and 'Close' in data.columns:
            rate = safe_float(data['Close'].iloc[-1]) / 100.0
            if rate is not None and 0 <= rate <= 0.2: # Sanity check
                if verbose: print(f"Using rate: {rate:.4f} ({rate*100:.2f}%)")
                return rate
            else: if verbose: print(f"Warn: Rate {rate} unusual.")
        else: if verbose: print("Warn: Could not fetch ^TNX data.")
    except Exception as e:
        if verbose: print(f"Error fetching rate: {e}")
        if debug_mode: traceback.print_exc()

    final_rate = safe_float(default_rate, 0.04) # Ensure default is float
    if verbose: print(f"Using default rate: {final_rate:.4f} ({final_rate*100:.2f}%)")
    return final_rate