# main_console.py
# Console interface for the Options Analyzer

import sys
import traceback
import numpy as np
import pandas as pd # Keep pandas import if used for display formatting
from tabulate import tabulate

# Import from other modules
from options_analyzer_core import OptionsAnalyzer
from utils import format_currency, clear_screen, safe_float

class ConsoleApp:
    """Handles console menu, input, and output formatting."""

    def __init__(self):
        self.analyzer = OptionsAnalyzer()
        # Ensure rate is fetched verbosely for console start
        self.analyzer.get_risk_free_rate(verbose=True)

    def display_main_menu(self):
        clear_screen(); print("+" + "="*35 + "+"); print("|     Options Analyzer Menu         |"); print("+" + "="*35 + "+")
        price_disp="N/A"
        if self.analyzer.current_ticker and self.analyzer.current_stock_data:
             price_disp = format_currency(self.analyzer.current_stock_data['current_price'], self.analyzer.current_stock_data.get('currency','USD'))
             print(f" Current: {self.analyzer.current_ticker} ({price_disp})")
        else: print(" Current: None")
        print("-" * 37); print("  1. Fetch Data"); print("  2. Simple Price"); print("  3. Options Chain"); print("  4. Analyze Strategy"); print("  5. Manage Favs"); print("  6. Settings"); print("  0. Exit"); print("-" * 37)
        if self.analyzer.favorite_tickers:
            fav_str = ", ".join(self.analyzer.favorite_tickers[:5])
            if len(self.analyzer.favorite_tickers) > 5: fav_str += "..."
            print(f" Favs: {fav_str}")
        print("+" + "="*35 + "+")

    def _get_ticker_input(self):
        """Handles getting ticker input, suggesting current/favs."""
        prompt="Ticker"; opts=[]; curr=self.analyzer.current_ticker
        if curr: opts.append(f"Enter='{curr}'")
        if self.analyzer.favorite_tickers: opts.append("'fav'=list")
        if opts: prompt += f" ({', '.join(opts)})"
        ticker_in = input(f"{prompt}: ").strip().upper()
        sel_ticker = None
        if not ticker_in and curr: sel_ticker = curr; print(f"Refresh {curr}...")
        elif ticker_in=='FAV' and self.analyzer.favorite_tickers:
            print("Favs:"); [print(f" {i+1}. {t}") for i,t in enumerate(self.analyzer.favorite_tickers)]
            fav_in = input("Num: ").strip()
            try:
                idx = int(fav_in)-1
                if 0 <= idx < len(self.analyzer.favorite_tickers): sel_ticker = self.analyzer.favorite_tickers[idx]
                else: print("Invalid selection.")
            except ValueError: print("Invalid number.")
        elif ticker_in: sel_ticker = ticker_in

        if sel_ticker:
            self.analyzer.get_stock_data(sel_ticker) # Fetch/refresh
        elif not ticker_in and not curr: print("No ticker entered.")

    def _get_expiration_input(self):
        """Handles getting expiration date input."""
        if not self.analyzer.current_stock_data: print("Fetch data first."); return None
        exps = self.analyzer.current_stock_data.get('expirations', ())
        if not exps: print("No expirations found for current ticker."); return None

        print("\nAvailable expiration dates:")
        valid_exps = []
        today = dt.datetime.now().date()
        for i, date_str in enumerate(exps):
            try:
                 exp_dt = dt.datetime.strptime(date_str, '%Y-%m-%d').date(); days = (exp_dt-today).days
                 if days >= 0: print(f"{len(valid_exps)+1}. {date_str} ({days}d)"); valid_exps.append({'date':date_str, 'days':days})
            except ValueError: continue # Skip bad dates silently
        if not valid_exps: print("No valid future dates."); return None

        while True:
            try:
                 sel = input(f"\nSelect date (1-{len(valid_exps)}), Enter=first: ").strip()
                 if not sel: selected = valid_exps[0]; print(f"Using: {selected['date']}"); break
                 idx = int(sel)-1
                 if 0 <= idx < len(valid_exps): selected = valid_exps[idx]; break
                 else: print("Invalid.")
            except ValueError: print("Invalid num.")
        return selected['date']

    def _get_strike_input(self):
        """Handles getting strike price input."""
        if not self.analyzer.current_stock_data: print("Fetch data first."); return None
        S = self.analyzer.current_stock_data['current_price']
        curr = self.analyzer.current_stock_data['currency']
        K = None
        while K is None:
            k_in = input(f"\nStrike (e.g. {S:.2f}) or 'atm': ").lower().strip()
            if k_in == 'atm':
                 # Simplified ATM finding for console - relies on core class state
                 try:
                     opts = self.analyzer.current_stock_data['ticker_object'].option_chain(self.analyzer.current_stock_data.get('_last_exp_selected', None)) # Needs last selected exp stored?
                     if not opts: raise ValueError("Need expiration") # Or re-prompt for expiration here?
                     strikes = sorted([s for s in pd.concat([opts.calls['strike'],opts.puts['strike']]).dropna().unique() if pd.notna(safe_float(s))])
                     if not strikes: print("No strikes found. Using price."); K = S
                     else: K = min(strikes, key=lambda x: abs(x - S)); print(f"ATM Strike: {format_currency(K, curr)}")
                 except Exception as e: print(f"ATM Err: {e}. Using price."); K = S # Fallback
            else:
                 k_val = safe_float(k_in)
                 if k_val is not None and k_val > 0: K = k_val
                 else: print("Invalid strike.")
        return K

    def run_simple_price_calculator(self):
        """Console workflow for simple price calculation."""
        if not self.analyzer.current_stock_data: print("Fetch data first (Opt 1)."); return
        s_data = self.analyzer.current_stock_data
        print(f"\nCalculate for {s_data['ticker']} ({format_currency(s_data['current_price'], s_data['currency'])})")

        exp_date = self._get_expiration_input()
        if not exp_date: return
        self.analyzer.current_stock_data['_last_exp_selected'] = exp_date # Store for potential ATM use

        strike = self._get_strike_input()
        if strike is None: return

        vol_hist = s_data['volatility']
        vol_use = vol_hist
        print(f"\nUsing {'Hist.' if pd.notna(vol_hist) else 'Default'} Vol: {vol_use*100:.2f}%" if pd.notna(vol_use) else "\nWarn: Hist Vol N/A.")
        if pd.isna(vol_use):
             while True:
                 try: u_vol=input("Est. vol [Enter=0.3]: ").strip(); vol_use=safe_float(u_vol,0.3);
                 if vol_use>0: break; elif u_vol=="": vol_use=0.3; break; else: print(">0 required")
                 except ValueError: print("Invalid num.")
             print(f"Using est. vol: {vol_use*100:.2f}%")

        option_type = None
        while option_type not in ['call','put','both']: opt_in=input("Type 'call','put','both' [both]: ").lower().strip() or 'both';
        if opt_in in ['call','put','both']: option_type=opt_in

        # Call the core calculation method
        results = self.analyzer.get_simple_option_price_data(strike, exp_date, option_type, vol_use)

        # Display results nicely
        if 'error' in results: print(f"Error: {results['error']}"); return

        p = results['params']
        o = results['output']
        print("\n--- Calculation Results ---")
        print(f"Params: S={format_currency(p['S'], p['currency'])}, K={format_currency(p['K'], p['currency'])}, T={p['T']:.4f} ({p['days']}d), "
              f"r={p['r']*100:.2f}%, q={p['q']*100:.2f}%, Vol={p['vol_used']*100:.2f}%")
        print("-" * 30)
        if 'call' in o:
            c = o['call']
            print(f"Call Option @ {format_currency(p['K'], p['currency'])}")
            print(f"  BSM Price: {format_currency(c['price'], p['currency'])}")
            if c['greeks'] and not all(np.isnan(v) for v in c['greeks'].values()):
                g=c['greeks']
                print(f"  Call Greeks: D={g['delta']:.4f}, G={g['gamma']:.4f}, T={format_currency(g['theta'], p['currency'])}/d, V={format_currency(g['vega'], p['currency'])}/1%, R={format_currency(g['rho'], p['currency'])}/1%")
            else: print("  Call Greeks: N/A")
            print("-" * 30)
        if 'put' in o:
            p_res = o['put'] # Renamed variable to avoid conflict
            print(f"Put Option @ {format_currency(p['K'], p['currency'])}")
            print(f"  BSM Price: {format_currency(p_res['price'], p['currency'])}")
            if p_res['greeks'] and not all(np.isnan(v) for v in p_res['greeks'].values()):
                g=p_res['greeks']
                print(f"  Put Greeks: D={g['delta']:.4f}, G={g['gamma']:.4f}, T={format_currency(g['theta'], p['currency'])}/d, V={format_currency(g['vega'], p['currency'])}/1%, R={format_currency(g['rho'], p['currency'])}/1%")
            else: print("  Put Greeks: N/A")
            print("-" * 30)

    def run_options_chain(self):
        """Console workflow for options chain."""
        if not self.analyzer.current_stock_data: print("Fetch data first (Opt 1)."); return
        s_data = self.analyzer.current_stock_data
        print(f"\nChain for {s_data['ticker']} ({format_currency(s_data['current_price'], s_data['currency'])})")

        exp_date = self._get_expiration_input()
        if not exp_date: return

        # Call the core data calculation method
        chain_df = self.analyzer.get_options_chain_data(exp_date)

        if chain_df is None or chain_df.empty: print("Could not retrieve/calculate chain data."); return

        # Display formatted table
        col_map = {'call_volume':'C Vol','call_oi':'C OI','market_call':'C Mkt','bsm_call':'C BSM','call_iv':'C IV%','call_iv_yf':'C IV%(YF)', 'call_delta':'C Del','call_gamma':'C Gam','call_theta':'C The','call_vega':'C Veg','call_rho':'C Rho', 'strike':'Strike', 'put_rho':'P Rho','put_vega':'P Veg','put_theta':'P The','put_gamma':'P Gam','put_delta':'P Del', 'put_iv':'P IV%','put_iv_yf':'P IV%(YF)','bsm_put':'P BSM','market_put':'P Mkt','put_oi':'P OI','put_volume':'P Vol'}
        base_cols_c=['call_volume','call_oi','market_call','bsm_call','call_iv']; base_cols_p=['put_iv','bsm_put','market_put','put_oi','put_volume']
        g_cols_c=['call_delta','call_gamma','call_theta','call_vega','call_rho']; g_cols_p=['put_rho','put_vega','put_theta','put_gamma','put_delta']
        disp_order = base_cols_c + (g_cols_c if self.analyzer.config['show_greeks_in_chain'] else []) + ['strike'] + (g_cols_p if self.analyzer.config['show_greeks_in_chain'] else []) + base_cols_p
        disp_df = chain_df[[c for c in disp_order if c in chain_df.columns]].copy(); disp_df.rename(columns=col_map, inplace=True)
        fmt_df = disp_df.copy(); curr = s_data['currency']
        for col in fmt_df.columns:
             if any(k in col for k in ['Mkt','BSM','Strike','The','Veg','Rho']): fmt_df[col] = fmt_df[col].apply(lambda x: format_currency(x, curr))
             elif 'IV%' in col: fmt_df[col] = fmt_df[col].apply(lambda x: f"{safe_float(x):.2f}%" if pd.notna(x) else 'N/A')
             elif any(k in col for k in ['Del','Gam']): fmt_df[col] = fmt_df[col].apply(lambda x: f"{safe_float(x):.4f}" if pd.notna(x) else 'N/A')
             elif any(k in col for k in ['Vol','OI']): fmt_df[col] = fmt_df[col].apply(lambda x: f"{int(safe_float(x,0)):,}" if pd.notna(x) else '0')

        print(f"\n--- Options Chain: {s_data['ticker']} | Exp: {exp_date} ---")
        # Add other parameters to header if needed
        tbl_str = tabulate(fmt_df, headers='keys', tablefmt='pretty', showindex=False, numalign="right", stralign="right")
        print("-" * min(150, len(tbl_str.split('\n')[0]))); print(tbl_str); print("-" * min(150, len(tbl_str.split('\n')[0])))

        # Ask to visualize
        viz_in = input("\nVisualize chain? (y/n): ").lower().strip()
        if viz_in == 'y':
             print("Generating plot...")
             fig = self.analyzer.generate_options_chain_plot(chain_df) # Use raw DF
             if fig is None: print("Failed to generate plot.")
             else:
                 try: plt.show() # Display the generated figure
                 except Exception as e: print(f"Plot display error: {e}")

    def run_strategy_analyzer(self):
        """Console workflow for strategy analysis."""
        if not self.analyzer.current_stock_data: print("Fetch data first (Opt 1)."); return
        s_data = self.analyzer.current_stock_data
        S0, curr = s_data['current_price'], s_data['currency']

        print("\n--- Strategy Analysis ---"); print("Select:"); strat_map = {1:"Covered Call", 2:"Protective Put", 3:"Bull Call Spread", 4:"Bear Put Spread", 5:"Long Straddle", 6:"Long Strangle"}
        [print(f" {i}. {n}") for i,n in strat_map.items()]; print(" 0. Back")
        strat_choice = None
        while strat_choice is None:
            try: choice = int(input("Num: ").strip());
            if choice==0: return; if choice in strat_map: strat_choice=choice; break; else: print("Invalid.")
            except ValueError: print("Invalid num.")

        exp_date = self._get_expiration_input()
        if not exp_date: return

        legs=[]; name=strat_map[strat_choice]; max_p_theory, max_l_theory=np.nan, np.nan; bes_theory=[]
        # Simplified local get_price - uses analyzer's cached chain if available
        def get_price(k, type):
            k_num=safe_float(k); if np.isnan(k_num) or k_num<=0: raise ValueError(f"Invalid K:{k}")
            opt_data = self.analyzer._get_option_data_for_strike(exp_date, k_num, type)
            price_num=np.nan
            if opt_data is not None:
                 bid,ask,last=safe_float(opt_data.get('bid'),0.0),safe_float(opt_data.get('ask'),0.0),safe_float(opt_data.get('lastPrice'))
                 if bid>0 and ask>bid: mid=(bid+ask)/2;
                 if pd.isna(last) or last<=0 or last<bid or last>ask: price_num=mid; print(f" Note: Mid used K={k_num}.")
                 else: price_num=last
                 elif pd.notna(last) and last>0: price_num=last; print(f" Note: Last used K={k_num}.")
            if np.isnan(price_num) or price_num<=0:
                 print(f"Warn: Mkt N/A K={k_num}. Using BSM."); # BSM fallback logic needed here, simplified for now
                 raise ValueError(f"Market price lookup failed for {type} K={k_num}") # Fail if market N/A for now
            return price_num

        try: # Wrap strategy specifics
            # (Strategy leg building logic - uses local get_price)
            # Simplified: Assumes get_price works or raises ValueError
            if strat_choice == 1: K_c=safe_float(input(f"Call K(>{S0:.2f}):").strip()); p_c=get_price(K_c,'call'); legs=[{'type':'stock','dir':'long','price':S0},{'type':'call','dir':'short','K':K_c,'price':p_c}]; cost=S0-p_c; bes_theory.append(cost); max_p_theory=K_c-cost; max_l_theory=-cost
            # ... (Add similar blocks for other strategies 2-6, using safe_float and get_price) ...
            # Example for Straddle [5]
            elif strat_choice == 5:
                 try: opts=s_data['ticker_object'].option_chain(exp_date); strikes=[s for s in pd.concat([opts.calls['strike'],opts.puts['strike']]).dropna().unique() if pd.notna(safe_float(s))]; K_atm=min(strikes,key=lambda x:abs(x-S0)); print(f"Using ATM K={format_currency(K_atm, curr)}")
                 except Exception as e: raise ValueError(f"ATM Err: {e}");
                 p_c=get_price(K_atm,'call'); p_p=get_price(K_atm,'put'); cost=p_c+p_p; legs=[{'type':'call','dir':'long','K':K_atm,'price':p_c},{'type':'put','dir':'long','K':K_atm,'price':p_p}]; max_p_theory=np.inf; max_l_theory=-cost; bes_theory.extend([K_atm-cost,K_atm+cost])

            if legs:
                payoff_data = self.analyzer.get_strategy_payoff_data(legs)
                if 'error' in payoff_data: print(f"Payoff Calc Error: {payoff_data['error']}"); return

                print("\n--- Strategy Summary ---"); print(f"Strat: {name} | Exp: {exp_date} | Curr Price: {format_currency(S0, curr)}"); print("Legs:")
                net_cost=0;
                for i,l in enumerate(legs): k_s=f" K={format_currency(l['K'], curr)}" if pd.notna(safe_float(l.get('K'))) else ""; p_s=f" @ {format_currency(l['price'], curr)}"; print(f"  {i+1}: {l['dir'].capitalize()} {l['type'].capitalize()}{k_s}{p_s}"); cost_leg = safe_float(l['price'],0); net_cost += cost_leg if l['dir']=='long' else -cost_leg
                print(f"Net {'Cost' if net_cost>1e-6 else 'Credit'}: {format_currency(abs(net_cost), curr)}") if abs(net_cost)>1e-6 else print("Net Cost: ~Zero")

                # Use theoretical BE/Max where available, else calculated
                be_disp = sorted(bes_theory if bes_theory else payoff_data['breakevens_calc'])
                max_p_disp = max_p_theory if pd.notna(max_p_theory) else payoff_data['max_profit_calc']
                max_l_disp = max_l_theory if pd.notna(max_l_theory) else payoff_data['max_loss_calc']
                be_s=", ".join([format_currency(b, curr) for b in be_disp]) or "N/A"; mp_s=format_currency(max_p_disp, curr) if np.isfinite(max_p_disp) else ('Unlim' if max_p_disp>0 else 'N/A'); ml_s=format_currency(max_l_disp, curr) if np.isfinite(max_l_disp) else ('Unlim' if max_l_disp<0 else 'N/A')
                print(f"\nBE(s)@Exp: {be_s}"); print(f"Max Profit: {mp_s}"); print(f"Max Loss: {ml_s}")

                print("\nGenerating Payoff...");
                fig = self.analyzer.generate_strategy_payoff_plot(payoff_data, name)
                if fig: plt.show()
                else: print("Failed to generate plot.")

        except ValueError as ve: print(f"\nInput/Calc Error: {ve}")
        except Exception as e: print(f"\nAnalysis Error: {e}"); if self.analyzer.config.get('debug_mode'): traceback.print_exc()

    # --- Main Console Loop ---
    def run(self):
        print("Welcome to Options Analyzer (Modular)!")
        while True:
            self.display_main_menu(); choice = input("Choice: ").strip()
            try:
                if choice == '1': self._get_ticker_input()
                elif choice == '2': self.run_simple_price_calculator()
                elif choice == '3': self.run_options_chain()
                elif choice == '4': self.run_strategy_analyzer()
                elif choice == '5': self.analyzer.manage_favorites() # Call method on analyzer instance
                elif choice == '6': self.analyzer.manage_settings()  # Call method on analyzer instance
                elif choice == '0': print("\nExit."); break
                else: print("Invalid.")
            except Exception as e: print(f"\nMenu Err: {e}"); if self.analyzer.config.get('debug_mode'): traceback.print_exc()
            if choice != '0': input("\nEnter to continue...")

# --- Entry Point ---
if __name__ == "__main__":
    console_ui = ConsoleApp()
    console_ui.run()