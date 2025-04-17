# options_analyzer_core.py
# Core OptionsAnalyzer class logic

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import json
import os
import traceback
import matplotlib.pyplot as plt
from matplotlib import style, ticker as mticker
from tabulate import tabulate

# Import from other modules
from utils import safe_float, format_currency
import options_formulas as formulas # Use functions from the formulas module

class OptionsAnalyzer:
    """
    Core class for fetching data, managing state, and performing analysis.
    Uses standalone functions for BSM/Greeks/IV calculations.
    Designed to be usable by console or UI frontends.
    """
    def __init__(self, config_path='options_config.json', fav_path='favorite_tickers.json'):
        self.current_ticker = None
        self.current_stock_data = None
        self.risk_free_rate = None
        self.config_path = config_path
        self.fav_path = fav_path
        self.config = self._load_config()
        self.favorite_tickers = self._load_favorite_tickers()
        self._chain_cache = {}
        # Load initial rate silently; explicit call needed for verbose console init
        self._fetch_risk_free_rate_silently() # Use silent fetch for general init

    # --- Config & Favorites Persistence ---
    def _load_config(self):
        default_config = {'volatility_days': 252, 'default_risk_free_rate': 0.04, 'show_greeks_in_chain': True,
                          'max_strikes_chain': 20, 'iv_precision': 0.0001, 'iv_max_iterations': 100,
                          'strategy_price_range': 0.3, 'debug_mode': False}
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f: config_data = json.load(f)
                config = default_config.copy(); config.update(config_data); print("Config loaded.")
                return config
            else: print("No config file, using defaults."); return default_config.copy()
        except Exception as e: print(f"Config load err: {e}. Using defaults."); return default_config.copy()

    def _save_config(self):
        try:
            with open(self.config_path, 'w') as f: json.dump(self.config, f, indent=4)
            print("Config saved.")
        except Exception as e: print(f"Config save err: {e}")

    def _load_favorite_tickers(self):
        try:
            if os.path.exists(self.fav_path):
                with open(self.fav_path, 'r') as f: favs = json.load(f)
                if isinstance(favs, list): print(f"Loaded {len(favs)} favs."); return favs
                else: print("Warn: Favs file bad fmt."); return []
            else: print("No favs file."); return []
        except Exception as e: print(f"Favs load err: {e}."); return []

    def _save_favorite_tickers(self):
        try:
            with open(self.fav_path, 'w') as f: json.dump(self.favorite_tickers, f, indent=4)
            print("Favs saved.")
        except Exception as e: print(f"Favs save err: {e}")

    # --- Data Validation & Fetching ---
    def validate_ticker(self, ticker, verbose=True):
        # (Same validation logic as before, potentially controlled by verbose flag)
        if not ticker or not isinstance(ticker, str): print("Ticker must be string."); return False
        ticker = ticker.upper().strip()
        try:
            if verbose: print(f"\nValidating '{ticker}'...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if hist.empty:
                 info = stock.info
                 if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
                      if verbose: print(f"'{ticker}' invalid/no data."); return False
            else: info = stock.info # Get info if history worked
            q_type = info.get('quoteType', 'N/A')
            allowed_types = ['EQUITY', 'ETF', 'INDEX', 'CURRENCY', 'COMMODITY']
            if q_type not in allowed_types and verbose: print(f"Warn: Type '{q_type}'. Options may differ.")
            if verbose: print(f"'{ticker}' valid ({info.get('shortName', 'N/A')}).")
            return True
        except Exception as e:
            if verbose: print(f"Validate err '{ticker}': {e}")
            if self.config.get('debug_mode'): traceback.print_exc(); return False


    def _fetch_risk_free_rate_silently(self):
        # Fetches rate without printing, returns the rate value
        try:
            tnx = yf.Ticker("^TNX"); data = tnx.download(period="5d", progress=False)
            if not data.empty and 'Close' in data.columns:
                rate = safe_float(data['Close'].iloc[-1]) / 100.0
                if rate is not None and 0 <= rate <= 0.2: self.risk_free_rate = rate; return rate
        except Exception: pass # Ignore errors in silent mode
        default = safe_float(self.config.get('default_risk_free_rate', 0.04), 0.04)
        self.risk_free_rate = default; return default

    def get_risk_free_rate(self, verbose=False):
        # Public method, potentially verbose
        fetched_rate = self._fetch_risk_free_rate_silently()
        if verbose:
            if fetched_rate == self.config.get('default_risk_free_rate'): print(f"Using default rate: {fetched_rate:.4f}")
            else: print(f"Using current rate: {fetched_rate:.4f}")
        return fetched_rate

    def get_stock_data(self, ticker):
        # (Same core data fetching logic as in previous corrected version)
        # Uses safe_float and format_currency from utils
        if not isinstance(ticker, str): return None
        ticker = ticker.upper().strip()
        # Use internal validation with verbose=True for console feedback
        if not self.validate_ticker(ticker, verbose=True): return None
        try:
            print(f"\nFetching data for {ticker}..."); stock = yf.Ticker(ticker)
            hist = stock.history(period="1y", interval="1d")
            price, vol = None, None
            if not hist.empty and 'Close' in hist.columns:
                hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce'); hist.dropna(subset=['Close'], inplace=True)
                if not hist.empty:
                    price = safe_float(hist['Close'].iloc[-1])
                    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
                    if len(returns) >= 10: vol = safe_float(returns.std() * np.sqrt(self.config['volatility_days']))
                    else: print(f"Warn: Only {len(returns)} returns for vol calc.")
            else: print(f"Warn: History issue for {ticker}.")
            info = {}; name, sector, industry, mcap_str, currency = ticker, 'N/A', 'N/A', 'N/A', 'USD'; div_yield = 0.0
            try:
                info = stock.info; name = info.get('shortName', ticker); sector = info.get('sector', 'N/A'); industry = info.get('industry', 'N/A')
                mcap = safe_float(info.get('marketCap')); currency = info.get('currency', 'USD'); div_yield = safe_float(info.get('dividendYield', 0.0), 0.0)
                if price is None or np.isnan(price):
                     for k in ['regularMarketPrice', 'currentPrice', 'previousClose', 'regularMarketOpen']:
                         p_val = safe_float(info.get(k))
                         if p_val is not None and not np.isnan(p_val) and p_val > 0: price = p_val; print(f"Using price from '{k}'"); break
                     if price is None or np.isnan(price):
                         hi, lo = safe_float(info.get('dayHigh')), safe_float(info.get('dayLow'))
                         if pd.notna(hi) and pd.notna(lo) and hi >= lo > 0: price = (hi+lo)/2; print("Warn: Using day hi/lo midpoint.")
                if mcap is not None and not np.isnan(mcap):
                    if mcap >= 1e12: mcap_str = f"{format_currency(mcap / 1e12, currency)}T"
                    elif mcap >= 1e9: mcap_str = f"{format_currency(mcap / 1e9, currency)}B"
                    elif mcap >= 1e6: mcap_str = f"{format_currency(mcap / 1e6, currency)}M"
                    else: mcap_str = format_currency(mcap, currency)
            except Exception as e_info: print(f"Warn: Info fetch issue: {e_info}");
            if price is None or np.isnan(price): raise ValueError(f"CRITICAL: Failed to find price for {ticker}.")
            print(f"\n=== {name} ({ticker}) ==="); print(f"Price: {format_currency(price, currency)}")
            print(f"Sector: {sector} | Industry: {industry}"); print(f"Market Cap: {mcap_str}")
            print(f"Dividend Yield: {div_yield:.4f} ({div_yield*100:.2f}%)")
            if vol is not None and not np.isnan(vol): print(f"Volatility (1y): {vol:.4f} ({vol*100:.2f}%)")
            else: print("Volatility (1y): N/A")
            expirations = ();
            try: expirations = stock.options; print(f"Found {len(expirations)} expirations.") if expirations else print("Note: No options found.")
            except Exception as e: print(f"Warn: Opt fetch issue: {e}")
            self.current_ticker = ticker
            self.current_stock_data = {'ticker': ticker, 'current_price': price, 'volatility': vol, 'dividend_yield': div_yield,
                                       'expirations': expirations, 'ticker_object': stock, 'history': hist, 'info': info, 'currency': currency}
            print(f"Data fetch complete."); return self.current_stock_data
        except Exception as e:
            print(f"\nError fetching data for '{ticker}': {e}");
            if self.config.get('debug_mode'): traceback.print_exc();
            if self.current_ticker == ticker: self.current_ticker, self.current_stock_data = None, None; return None

    def _get_option_data_for_strike(self, expiration_date, strike, option_type):
        # (Same logic as previous corrected version, uses safe_float)
         if not self.current_stock_data: print("Err: Stock data missing."); return None
         if not expiration_date: print("Err: Date required."); return None
         strike_num = safe_float(strike); option_type = option_type.lower()
         if np.isnan(strike_num) or strike_num <= 0: print("Err: Invalid strike."); return None
         stock = self.current_stock_data['ticker_object']; cache_key = f"{expiration_date}"
         try:
             if cache_key not in self._chain_cache or self._chain_cache[cache_key]['date'] != expiration_date:
                 if self.config.get('debug_mode'): print(f"Cache miss. Fetching {expiration_date}...")
                 opt_chain = stock.option_chain(expiration_date)
                 self._chain_cache[cache_key] = {'date': expiration_date, 'calls': opt_chain.calls, 'puts': opt_chain.puts}
             data_df = self._chain_cache[cache_key]['calls' if option_type == 'call' else 'puts']
             if 'strike' not in data_df.columns: return None
             data_df['strike'] = pd.to_numeric(data_df['strike'], errors='coerce')
             option_series = data_df[data_df['strike'] == strike_num]
             return option_series.iloc[0] if not option_series.empty else None
         except IndexError: print(f"Err: No options structure for {expiration_date}."); self._chain_cache.pop(cache_key, None); return None
         except Exception as e:
             print(f"Err fetch K={strike_num} ({option_type}): {e}");
             if self.config.get('debug_mode'): traceback.print_exc(); self._chain_cache.pop(cache_key, None); return None

    # --- Analysis Methods ---
    def get_simple_option_price_data(self, strike, expiration_date, option_type, use_volatility=None):
        """
        Calculates BSM price and Greeks for a single option.
        Returns a dictionary with results, suitable for UI or console display.
        Does NOT use input() or print(). Requires stock data to be loaded.
        """
        if not self.current_stock_data: return {'error': "Stock data not loaded."}
        s_data = self.current_stock_data
        S, vol_hist, q, curr, r = s_data['current_price'], s_data['volatility'], s_data['dividend_yield'], s_data['currency'], self.risk_free_rate

        # Determine volatility to use
        vol_to_use = use_volatility # Use provided vol if available
        if vol_to_use is None or np.isnan(vol_to_use): vol_to_use = vol_hist # Fallback to historical
        if vol_to_use is None or np.isnan(vol_to_use): vol_to_use = 0.3 # Final fallback to default
        vol_to_use = safe_float(vol_to_use, default=0.3) # Ensure numeric

        # Validate other inputs
        K = safe_float(strike);
        if np.isnan(K) or K <= 0: return {'error': "Invalid strike price."}
        if r is None: r = self._fetch_risk_free_rate_silently() # Ensure rate exists

        try:
            today = dt.datetime.now().date(); exp_dt = dt.datetime.strptime(expiration_date, '%Y-%m-%d').date()
            days = max(0, (exp_dt - today).days); T = days / 365.0
        except ValueError: return {'error': "Invalid expiration date format."}

        results = {'params': {'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'vol_used': vol_to_use, 'currency': curr, 'exp_date': expiration_date, 'days': days}}
        output = {}

        if option_type.lower() in ['call', 'both']:
            bsm_price = formulas.black_scholes_merton(S, K, T, r, q, vol_to_use, "call")
            greeks = formulas.calculate_option_greeks(S, K, T, r, q, vol_to_use, "call")
            output['call'] = {'price': bsm_price, 'greeks': greeks}

        if option_type.lower() in ['put', 'both']:
            bsm_price = formulas.black_scholes_merton(S, K, T, r, q, vol_to_use, "put")
            greeks = formulas.calculate_option_greeks(S, K, T, r, q, vol_to_use, "put")
            output['put'] = {'price': bsm_price, 'greeks': greeks}

        results['output'] = output
        return results

    def get_options_chain_data(self, expiration_date):
        """
        Calculates and returns the raw options chain data as a DataFrame.
        Requires stock data to be loaded. Handles calculations internally.
        Minimal printing, designed for programmatic use.
        """
        if not self.current_stock_data: print("Err: Stock data missing."); return None
        s_data = self.current_stock_data
        S, vol_hist, q, exps, stock, curr, r = s_data['current_price'], s_data['volatility'], s_data['dividend_yield'], s_data['expirations'], s_data['ticker_object'], s_data['currency'], self.risk_free_rate
        target_ticker = s_data['ticker']

        if not expiration_date or (exps and expiration_date not in exps): print(f"Err: Invalid/missing expiration '{expiration_date}'."); return None
        if not exps and expiration_date: print(f"Warn: No expirations listed, attempting fetch for {expiration_date} anyway.") # Allow attempt if user provides date

        vol_use = vol_hist if pd.notna(vol_hist) else 0.3 # Default vol if hist missing
        if r is None: r = self._fetch_risk_free_rate_silently()

        try: today=dt.datetime.now().date(); exp_dt=dt.datetime.strptime(expiration_date,'%Y-%m-%d').date(); days=max(0,(exp_dt-today).days); T=days/365.0
        except ValueError: print(f"Err: Bad date format '{expiration_date}'."); return None

        try:
            print(f"Fetching chain data for {target_ticker} ({expiration_date})...")
            self._chain_cache = {} # Clear old cache
            opts = stock.option_chain(expiration_date); calls, puts = opts.calls, opts.puts; self._chain_cache[expiration_date] = {'date':expiration_date,'calls':calls,'puts':puts}
            if calls.empty and puts.empty: print("No options data found."); return pd.DataFrame()

            calls = calls.add_prefix('c_').rename(columns={'c_strike':'strike'}); puts = puts.add_prefix('p_').rename(columns={'p_strike':'strike'})
            num_cols = ['strike', 'c_lastPrice', 'c_bid', 'c_ask', 'c_volume', 'c_openInterest', 'c_impliedVolatility', 'p_lastPrice', 'p_bid', 'p_ask', 'p_volume', 'p_openInterest', 'p_impliedVolatility']
            df = pd.merge(calls, puts, on='strike', how='outer')
            for col in num_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.sort_values(by='strike', inplace=True); df.reset_index(drop=True, inplace=True); df.dropna(subset=['strike'], inplace=True)

            max_K = self.config['max_strikes_chain']
            if len(df) > max_K and pd.notna(S):
                 df['k_diff']=(df['strike']-S).abs(); atm_pos=df.index.get_loc(df['k_diff'].idxmin())
                 h_w=max_K//2; start=max(0,atm_pos-h_w); end=min(len(df),start+max_K);
                 if(end-start)<max_K: start=max(0,end-max_K); df=df.iloc[start:end].reset_index(drop=True)

            results = []
            print("Calculating BSM/Greeks/IV for chain...")
            for _, row in df.iterrows():
                # (Calculation logic identical to previous corrected calculate_options_chain)
                # Uses formulas.black_scholes_merton, formulas.calculate_option_greeks, etc.
                K = row['strike']; dr = {'strike': K}
                c_b, c_a, c_l = row.get('c_bid'), row.get('c_ask'), row.get('c_lastPrice')
                p_b, p_a, p_l = row.get('p_bid'), row.get('p_ask'), row.get('p_lastPrice')
                c_mkt = c_l; p_mkt = p_l
                if c_b>0 and c_a>c_b: mid=(c_b+c_a)/2; if pd.isna(c_l) or c_l<=0 or c_l<c_b or c_l>c_a: c_mkt=mid
                if p_b>0 and p_a>p_b: mid=(p_b+p_a)/2; if pd.isna(p_l) or p_l<=0 or p_l<p_b or p_l>p_a: p_mkt=mid
                dr['market_call'], dr['market_put'] = c_mkt, p_mkt
                iv_c = formulas.calculate_implied_volatility(S, K, T, r, q, c_mkt, "call", self.config['iv_precision'], self.config['iv_max_iterations']); dr['call_iv'] = iv_c*100 if pd.notna(iv_c) else np.nan
                iv_p = formulas.calculate_implied_volatility(S, K, T, r, q, p_mkt, "put", self.config['iv_precision'], self.config['iv_max_iterations']); dr['put_iv'] = iv_p*100 if pd.notna(iv_p) else np.nan
                dr['call_iv_yf'], dr['put_iv_yf'] = row.get('c_impliedVolatility',np.nan)*100, row.get('p_impliedVolatility',np.nan)*100
                iv_yf_c, iv_yf_p = row.get('c_impliedVolatility'), row.get('p_impliedVolatility')
                vol_c = iv_c if pd.notna(iv_c) else (iv_yf_c if pd.notna(iv_yf_c) else vol_use) # Prioritize calculated IV
                vol_p = iv_p if pd.notna(iv_p) else (iv_yf_p if pd.notna(iv_yf_p) else vol_use)
                dr['bsm_call']=formulas.black_scholes_merton(S, K, T, r, q, vol_c, "call"); dr['bsm_put']=formulas.black_scholes_merton(S, K, T, r, q, vol_p, "put")
                if self.config['show_greeks_in_chain']:
                    g_c=formulas.calculate_option_greeks(S, K, T, r, q, vol_c, "call"); dr.update({f'call_{k}':v for k,v in g_c.items()})
                    g_p=formulas.calculate_option_greeks(S, K, T, r, q, vol_p, "put"); dr.update({f'put_{k}':v for k,v in g_p.items()})
                dr['call_volume'], dr['call_oi'] = row.get('c_volume'), row.get('c_openInterest')
                dr['put_volume'], dr['put_oi'] = row.get('p_volume'), row.get('p_openInterest')
                results.append(dr)
            print("Chain processing complete.")
            return pd.DataFrame(results) if results else pd.DataFrame()

        except AttributeError as ae: print(f"Chain Attr Err: {ae}"); return None
        except Exception as e: print(f"Chain Calc Err: {e}"); if self.config.get('debug_mode'): traceback.print_exc(); return None


    def get_strategy_payoff_data(self, strategy_legs):
        """
        Calculates payoff data for a given strategy definition.
        Requires stock data to be loaded (for S0).
        Returns dictionary with PnL array, S_T range, max profit/loss, breakevens.
        """
        if not self.current_stock_data: return {'error': "Stock data not loaded."}
        if not strategy_legs: return {'error': "Strategy legs cannot be empty."}

        S0 = self.current_stock_data['current_price']
        currency = self.current_stock_data['currency']

        # --- Calculate Max Profit/Loss and Breakevens (Simplified examples) ---
        # Note: Accurate calculation depends heavily on the strategy structure.
        # This requires more complex logic specific to each strategy type.
        # For now, we'll calculate payoff range and let plotting handle limits.
        # A more robust implementation would identify strategy type and apply formulas.
        max_profit_est, max_loss_est = np.nan, np.nan # Placeholder
        breakevens_est = [] # Placeholder

        # --- Define Payoff Range ---
        try:
            crit_points = [S0] + [safe_float(leg.get('K')) for leg in strategy_legs if pd.notna(safe_float(leg.get('K')))] # + breakevens_est
            valid_points = [p for p in crit_points if pd.notna(p)];
            if not valid_points: valid_points = [S0] if pd.notna(S0) else [100] # Need some point

            price_range_factor = self.config['strategy_price_range']
            S_T_min = max(0, min(valid_points) * (1 - price_range_factor * 1.5))
            S_T_max = max(valid_points) * (1 + price_range_factor * 1.5)
            if S_T_max <= S_T_min: S_T_max = S_T_min * 1.1 # Ensure range is valid
            S_T_range = np.linspace(S_T_min, S_T_max, 150)

            # --- Calculate P/L ---
            # Define internal payoff calc (uses safe_float)
            def _calculate_payoff_internal(S_T, legs):
                S_T_num = safe_float(S_T); cost=0; payoff=0
                if np.isnan(S_T_num): return np.nan
                for leg in legs:
                    price = safe_float(leg.get('price'),0.0); K=safe_float(leg.get('K')); dir, l_type = leg.get('dir','long'), leg.get('type')
                    cost += price if dir=='long' else -price
                    leg_p=0
                    if l_type=='stock': leg_p=S_T_num
                    elif l_type=='call' and not np.isnan(K): leg_p=max(0.0, S_T_num-K)
                    elif l_type=='put' and not np.isnan(K): leg_p=max(0.0, K-S_T_num)
                    payoff += leg_p if dir=='long' else -leg_p
                return payoff - cost

            PnL = np.array([_calculate_payoff_internal(s_t, strategy_legs) for s_t in S_T_range])

            # Basic estimation of max/min from calculated range (not theoretical max/min)
            if PnL.size > 0:
                max_profit_est = np.nanmax(PnL)
                max_loss_est = np.nanmin(PnL)

            # Simple Breakeven Estimation (crossing zero) - very basic
            sign_change = np.where(np.diff(np.sign(PnL)))[0]
            for idx in sign_change:
                # Interpolate between points where sign changes
                if abs(PnL[idx+1] - PnL[idx]) > 1e-9: # Avoid division by zero if PnL is flat
                    be = S_T_range[idx] - PnL[idx] * (S_T_range[idx+1] - S_T_range[idx]) / (PnL[idx+1] - PnL[idx])
                    breakevens_est.append(be)

            return {
                'S_T_range': S_T_range,
                'PnL': PnL,
                'max_profit_calc': max_profit_est, # Calculated max in range
                'max_loss_calc': max_loss_est,     # Calculated min in range
                'breakevens_calc': sorted(list(set(breakevens_est))), # Calculated BEs in range
                'currency': currency
            }

        except Exception as e:
            print(f"Strategy payoff calc error: {e}")
            if self.config.get('debug_mode'): traceback.print_exc()
            return {'error': f"Calculation error: {e}"}


    # --- Plotting ---
    def _create_plot_figure(self, title, x_label, y_label, size=(10, 6)):
        """Helper to create a matplotlib figure and axes object."""
        fig, ax = plt.subplots(figsize=size); fig.patch.set_facecolor('white'); ax.set_facecolor('#f0f0f0')
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel(x_label, fontsize=11); ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle=':', linewidth=0.5, color='grey')
        ax.tick_params(axis='both', which='major', labelsize=10)
        return fig, ax

    def generate_options_chain_plot(self, chain_df):
        """
        Generates the options chain visualization (Prices and IV).
        Requires a DataFrame from get_options_chain_data. Returns matplotlib fig.
        """
        if chain_df is None or chain_df.empty: return None
        if not self.current_stock_data: return None # Need S0, currency etc.
        S0, currency, ticker, vol_hist = self.current_stock_data['current_price'], self.current_stock_data['currency'], self.current_stock_data['ticker'], self.current_stock_data['volatility']
        exp_date = "Unknown Exp" # Need expiration date for title - should be passed or stored

        # --- Plotting Setup ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True); fig.patch.set_facecolor('white')
        fig.suptitle(f"{ticker} Options Chain ({exp_date}) | Price: {format_currency(S0, currency)}", fontsize=15, weight='bold')

        # Plot 1: Prices
        ax1 = axes[0]; ax1.set_facecolor('#f0f0f0'); ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{format_currency(1, currency)[0]}%.2f'))
        plots_p1 = {'market_call':'bo-','bsm_call':'c--','market_put':'ro-','bsm_put':'m--'}
        labels_p1 = {'market_call':'Market Call','bsm_call':'BSM Call','market_put':'Market Put','bsm_put':'BSM Put'}
        for col, style in plots_p1.items():
            if col in chain_df.columns: data=pd.to_numeric(chain_df[col],errors='coerce');
            if data.notna().any(): ax1.plot(chain_df['strike'], data, style, label=labels_p1[col], markersize=4 if '-' in style else None, alpha=0.8)
        ax1.set_ylabel(f'Price ({currency})'); ax1.set_title('Market vs BSM Prices'); ax1.grid(True, ls=':'); ax1.axvline(S0, color='k', ls=':', lw=1.5, label='Current Price'); ax1.legend(fontsize=9)

        # Plot 2: IV
        ax2 = axes[1]; ax2.set_facecolor('#f0f0f0'); ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        plots_p2 = {'call_iv':'g^-','put_iv':'yv-','call_iv_yf':'c:','put_iv_yf':'m:'}
        labels_p2 = {'call_iv':'Call IV(Calc)','put_iv':'Put IV(Calc)','call_iv_yf':'Call IV(YF)','put_iv_yf':'Put IV(YF)'}
        valid_ivs = []
        for col, style in plots_p2.items():
            if col in chain_df.columns: data=pd.to_numeric(chain_df[col],errors='coerce');
            if data.notna().any(): ax2.plot(chain_df['strike'], data, style, label=labels_p2[col], markersize=4 if '-' in style else None, alpha=0.7); valid_ivs.extend(data.dropna().tolist())
        if pd.notna(vol_hist): hv_pct=vol_hist*100; ax2.axhline(hv_pct, color='dimgray', ls='--', lw=1.5, label=f'HistVol({hv_pct:.1f}%)'); valid_ivs.append(hv_pct)
        ax2.set_xlabel('Strike'); ax2.set_ylabel('Implied Vol (%)'); ax2.set_title('IV Smile/Skew'); ax2.grid(True, ls=':'); ax2.axvline(S0, color='k', ls=':', lw=1.5); ax2.legend(fontsize=9)
        if valid_ivs: min_iv,max_iv = min(valid_ivs),max(valid_ivs); pad=max((max_iv-min_iv)*0.1, 5.0); ax2.set_ylim(max(0,min_iv-pad), max_iv+pad)

        plt.tight_layout(rect=[0,0.03,1,0.95]); return fig

    def generate_strategy_payoff_plot(self, payoff_data, strategy_name="Strategy"):
        """
        Generates the strategy payoff plot.
        Requires payoff data from get_strategy_payoff_data. Returns matplotlib fig.
        """
        if not payoff_data or 'error' in payoff_data: return None
        S_T, PnL, max_p, max_l, bes, curr = payoff_data['S_T_range'], payoff_data['PnL'], payoff_data['max_profit_calc'], payoff_data['max_loss_calc'], payoff_data['breakevens_calc'], payoff_data['currency']

        fig, ax = self._create_plot_figure(f"{strategy_name} Payoff", f"Price@Exp ({curr})", f"Profit/Loss ({curr})", size=(11, 6.5))
        sym = format_currency(1, curr)[0] if curr else '$'; ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{sym}%.2f')); ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'{sym}%.2f'))

        ax.plot(S_T, PnL, lw=2.5, color='navy', label='P/L'); ax.axhline(0, color='k', ls='--', lw=1, label='BE Level')
        if bes:
            ax.scatter(bes, [0]*len(bes), c='r', s=100, zorder=5, edgecolors='k', label='BE(s)')
            y_r=max(abs(PnL.min()), abs(PnL.max())) if len(PnL)>0 else 1; offset=y_r*0.05 if y_r!=0 else 0.1
            for be in bes: ax.text(be, offset, f'BE: {format_currency(be, curr)}', c='darkred', ha='center', va='bottom', fontsize=9, weight='bold')

        p_lbl=f'Calc Max Profit: {format_currency(max_p, curr)}' if pd.notna(max_p) else 'Calc Max Profit: N/A'
        l_lbl=f'Calc Min Loss: {format_currency(max_l, curr)}' if pd.notna(max_l) else 'Calc Min Loss: N/A'
        # Theoretical max/min require strategy-specific logic - using calculated range min/max here
        ax.axhline(max_p, c='g', ls=':', lw=1.5, label=p_lbl) if pd.notna(max_p) else None
        ax.axhline(max_l, c='r', ls=':', lw=1.5, label=l_lbl) if pd.notna(max_l) else None

        y_min, y_max = np.nanmin(PnL) if len(PnL)>0 else -1, np.nanmax(PnL) if len(PnL)>0 else 1
        if pd.notna(max_l): y_min=min(y_min,max_l);
        if pd.notna(max_p): y_max=max(y_max,max_p);
        pad=(y_max-y_min)*0.1 if (y_max!=y_min) else 1.0; ax.set_ylim(y_min-pad, y_max+pad)
        ax.legend(fontsize=9); plt.tight_layout(); return fig

# End of options_analyzer_core.py