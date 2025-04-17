# options_analyzer_core.py
import numpy as np
import pandas as pd
import datetime as dt
import traceback
from tabulate import tabulate

# Import from other modules
import utils
import config_manager
import data_fetcher
import bsm_formulas
import plotting
import strategies

class OptionsAnalyzer:
    """ Orchestrates option analysis using modular functions. """

    def __init__(self):
        """Initialize the analyzer."""
        self.config = config_manager.load_config()
        self.favorite_tickers = config_manager.load_favorites()
        self.risk_free_rate = data_fetcher.get_risk_free_rate(
            default_rate=self.config.get('default_risk_free_rate', 0.04),
            verbose=True, # Verbose for console init
            debug_mode=self.config.get('debug_mode', False)
        )
        self.current_ticker = None
        self.current_stock_data = None
        self._chain_cache = {} # Cache for option chain data {exp_date: {'date':..., 'calls':df, 'puts':df}}

    def _is_debug(self):
        """ Check if debug mode is enabled in config. """
        return self.config.get('debug_mode', False)

    # --- Data Handling Methods ---
    def fetch_and_set_stock_data(self, ticker):
        """ Fetches data using data_fetcher and updates internal state. """
        stock_data = data_fetcher.get_stock_data(ticker, debug_mode=self._is_debug())
        if stock_data:
            self.current_ticker = stock_data['ticker']
            self.current_stock_data = stock_data
            self._chain_cache = {} # Clear chain cache when ticker changes
            return True
        else:
            # If fetch failed for the current ticker, reset it
            if self.current_ticker == ticker.upper().strip():
                self.current_ticker = None
                self.current_stock_data = None
            return False

    def _get_option_data_for_strike(self, expiration_date, strike):
         """ Helper to get option data rows (call & put) for a strike using cache. """
         # This method tightly couples with the class state (cache, stock_object)
         # Kept within the class for now.
         if not self.current_stock_data: return None, None
         stock_obj = self.current_stock_data.get('ticker_object')
         if not stock_obj: return None, None
         if not expiration_date: return None, None
         strike_num = utils.safe_float(strike)
         if np.isnan(strike_num) or strike_num <= 0: return None, None

         cache_key = expiration_date
         try:
             # Check cache
             if cache_key not in self._chain_cache or self._chain_cache[cache_key]['date'] != expiration_date:
                  if self._is_debug(): print(f"Cache miss/stale. Fetching {expiration_date}...")
                  opt_chain = stock_obj.option_chain(expiration_date)
                  # Store dataframes, ensuring strike is numeric
                  calls_df = opt_chain.calls
                  puts_df = opt_chain.puts
                  if 'strike' in calls_df.columns: calls_df['strike'] = pd.to_numeric(calls_df['strike'], errors='coerce')
                  if 'strike' in puts_df.columns: puts_df['strike'] = pd.to_numeric(puts_df['strike'], errors='coerce')
                  self._chain_cache[cache_key] = {'date': expiration_date, 'calls': calls_df, 'puts': puts_df}

             # Retrieve from cache
             calls_df = self._chain_cache[cache_key]['calls']
             puts_df = self._chain_cache[cache_key]['puts']

             # Find rows for the specific strike
             call_row = calls_df[calls_df['strike'] == strike_num].iloc[0] if 'strike' in calls_df.columns and not calls_df[calls_df['strike'] == strike_num].empty else None
             put_row = puts_df[puts_df['strike'] == strike_num].iloc[0] if 'strike' in puts_df.columns and not puts_df[puts_df['strike'] == strike_num].empty else None

             return call_row, put_row # Return Series or None

         except Exception as e:
             print(f"Error getting option data K={strike_num} for {expiration_date}: {e}")
             if self._is_debug(): traceback.print_exc()
             self._chain_cache.pop(cache_key, None) # Clear bad cache entry
             return None, None

    # --- Console Interaction Methods --- (Select Expiration)
    def _select_expiration_date_console(self):
        """ Console helper to select expiration date. """
        if not self.current_stock_data: return None
        expirations = self.current_stock_data.get('expirations', ())
        if not expirations: print("No expirations available."); return None

        print("\nAvailable expiration dates:")
        valid_expirations = []
        today = dt.datetime.now().date()
        for i, date_str in enumerate(expirations):
             try:
                 if isinstance(date_str, dt.date): exp_date, fmt_str = date_str, date_str.strftime('%Y-%m-%d')
                 elif isinstance(date_str, str): exp_date, fmt_str = dt.datetime.strptime(date_str, '%Y-%m-%d').date(), date_str
                 else: continue
                 days = (exp_date - today).days
                 if days >= 0:
                     print(f"{len(valid_expirations)+1}. {fmt_str} ({days} days)")
                     valid_expirations.append({'idx':i, 'date':fmt_str, 'days':days})
             except ValueError: continue # Skip invalid formats silently

        if not valid_expirations: print("No valid future dates."); return None

        while True:
            try:
                sel = input(f"\nSelect date (1-{len(valid_expirations)}), Enter=first: ").strip()
                if not sel: sel_data = valid_expirations[0]; print(f"Using: {sel_data['date']}"); break
                idx = int(sel) - 1
                if 0 <= idx < len(valid_expirations): sel_data = valid_expirations[idx]; break
                else: print("Invalid selection.")
            except ValueError: print("Invalid input.")
        print(f"\nSelected: {sel_data['date']} ({sel_data['days']} days)"); return sel_data['date']

    # --- Core Functionality ---

    def get_simple_option_price(self):
        """ Calculate and display simple BSM price and Greeks (Console). """
        if not self.current_stock_data: print("\nPlease fetch data (Opt 1)."); return
        s_data = self.current_stock_data
        S, vol, q, exps, curr, r = s_data['current_price'], s_data['volatility'], s_data['dividend_yield'], s_data['expirations'], s_data['currency'], self.risk_free_rate
        vol_use = vol
        if pd.isna(vol_use): # Check for NaN explicitly
             print("\nWarn: Vol unavailable/NaN.")
             while True:
                 try: u_vol_str = input("Est. vol (e.g. 0.3) [Enter=0.3]: ").strip(); vol_use = utils.safe_float(u_vol_str, 0.3)
                 if vol_use > 0: break; elif u_vol_str=="": vol_use=0.3; break; else: print(">0 required.")
                 except ValueError: print("Invalid number.")
             print(f"Using est. vol: {vol_use*100:.2f}%")
        if pd.isna(r): r = data_fetcher.get_risk_free_rate(self.config['default_risk_free_rate'], True, self._is_debug()) # Fetch if needed

        exp_date = self._select_expiration_date_console()
        if not exp_date: return
        today=dt.datetime.now().date(); exp_dt=dt.datetime.strptime(exp_date, '%Y-%m-%d').date(); days=max(0,(exp_dt-today).days); T=days/365.0
        print(f"T: {T:.4f} years ({days} days)")

        K = None # Get Strike
        while K is None:
             k_in = input(f"\nStrike (e.g. {S:.2f}) or 'atm': ").lower().strip()
             if k_in == 'atm':
                 try:
                    stock_obj = s_data.get('ticker_object')
                    if not stock_obj: raise ValueError("Ticker object missing")
                    opts = stock_obj.option_chain(exp_date)
                    strikes = sorted([s for s in pd.concat([opts.calls['strike'],opts.puts['strike']]).dropna().unique() if pd.notna(utils.safe_float(s))])
                    if not strikes: print("No strikes found. Using price."); K = S
                    else: K = min(strikes, key=lambda x: abs(x - S)); print(f"ATM Strike