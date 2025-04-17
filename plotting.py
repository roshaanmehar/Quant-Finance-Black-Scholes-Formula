# plotting.py
import matplotlib.pyplot as plt
from matplotlib import style, ticker as mticker
import numpy as np
import pandas as pd
from utils import safe_float, format_currency

style.use('seaborn-v0_8-darkgrid')

def visualize_options_chain(df, current_price, currency, expiration_date, ticker_symbol=""):
    """Visualize the options chain data. Uses matplotlib."""
    if df is None or df.empty: print("No data to visualize."); return
    if 'strike' not in df.columns: print("Err: Missing 'strike' column."); return
    current_price_num = safe_float(current_price)
    if pd.isna(current_price_num): print("Err: Invalid current price."); return

    df_vis = df.copy(); df_vis['strike'] = pd.to_numeric(df_vis['strike'], errors='coerce'); df_vis.dropna(subset=['strike'], inplace=True)
    if df_vis.empty: print("No valid strike data for viz."); return

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{ticker_symbol} Options Chain ({expiration_date})\nCurrent Price: {format_currency(current_price_num, currency)}", fontsize=15, weight='bold')
    fig.patch.set_facecolor('white')

    # Plot 1: Prices
    ax1 = axes[0]; ax1.set_facecolor('#f0f0f0'); ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{format_currency(1, currency)[0]}%.2f'))
    p1_cols = {'market_call': {'label': 'Market Call', 'style': 'bo-', 'alpha': 0.8, 'size': 5}, 'bsm_call': {'label': 'BSM Call', 'style': 'c--', 'alpha': 0.8, 'size': None}, 'market_put': {'label': 'Market Put', 'style': 'ro-', 'alpha': 0.8, 'size': 5}, 'bsm_put': {'label': 'BSM Put', 'style': 'm--', 'alpha': 0.8, 'size': None}}
    for col, p in p1_cols.items():
        if col in df_vis.columns: data = pd.to_numeric(df_vis[col], errors='coerce');
        if data.notna().any(): ax1.plot(df_vis['strike'], data, p['style'], label=p['label'], markersize=p['size'], alpha=p['alpha'])
    ax1.set_ylabel(f'Option Price ({currency})', fontsize=11); ax1.set_title('Market vs. Calculated (BSM) Prices', fontsize=13); ax1.grid(True, linestyle=':'); ax1.axvline(current_price_num, color='k', linestyle=':', lw=1.5, label=f'Current Price'); ax1.legend(fontsize=9); ax1.tick_params(labelsize=10)

    # Plot 2: IV
    ax2 = axes[1]; ax2.set_facecolor('#f0f0f0'); ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    p2_cols = {'call_iv': {'label': 'Call IV (Calc)', 'style': 'g^-', 'alpha': 0.8, 'size': 5}, 'put_iv': {'label': 'Put IV (Calc)', 'style': 'yv-', 'alpha': 0.8, 'size': 5}, 'call_iv_yf': {'label': 'Call IV (YF)', 'style': 'c:', 'alpha': 0.7, 'size': 4}, 'put_iv_yf': {'label': 'Put IV (YF)', 'style': 'm:', 'alpha': 0.7, 'size': 4}}
    valid_ivs = []
    for col, p in p2_cols.items():
        if col in df_vis.columns: data = pd.to_numeric(df_vis[col], errors='coerce');
        if data.notna().any(): ax2.plot(df_vis['strike'], data, p['style'], label=p['label'], markersize=p['size'], alpha=p['alpha']); valid_ivs.extend(data.dropna().tolist())
    # Note: Hist vol line removed as hist_vol is not directly passed; could be added if needed
    ax2.set_xlabel('Strike Price', fontsize=11); ax2.set_ylabel('Implied Volatility (%)', fontsize=11); ax2.set_title('Implied Volatility Smile / Skew', fontsize=13); ax2.grid(True, linestyle=':'); ax2.axvline(current_price_num, color='k', ls=':', lw=1.5); ax2.legend(fontsize=9); ax2.tick_params(labelsize=10)
    if valid_ivs: min_iv, max_iv = min(valid_ivs), max(valid_ivs); pad = max((max_iv-min_iv)*0.1, 5.0); ax2.set_ylim(max(0,min_iv-pad), max_iv+pad)

    plt.tight_layout(rect=[0,0.03,1,0.95]);
    try: plt.show()
    except Exception as e: print(f"\nPlot display Err: {e}.")

def plot_payoff_diagram(S_T_range, PnL, strategy_name, breakevens, max_profit, max_loss, currency):
    """Plots the Profit/Loss diagram for a strategy."""
    if not isinstance(S_T_range, (np.ndarray, list)) or not isinstance(PnL, (np.ndarray, list)): return
    if len(S_T_range) != len(PnL): return
    max_p, max_l = safe_float(max_profit, np.inf), safe_float(max_loss, -np.inf)
    fig, ax = plt.subplots(figsize=(11, 6.5)); fig.patch.set_facecolor('white'); ax.set_facecolor('#f0f0f0')
    ax.plot(S_T_range, PnL, lw=2.5, color='navy', label='P/L'); ax.axhline(0, color='k', ls='--', lw=1, label='BE Level')
    sym = format_currency(1, currency)[0] if currency else '$'; ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'{sym}%.2f')); ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'{sym}%.2f')); ax.tick_params(labelsize=10)
    valid_bes = sorted([safe_float(be) for be in breakevens if pd.notna(safe_float(be))])
    if valid_bes:
        ax.scatter(valid_bes, [0] * len(valid_bes), c='r', s=100, zorder=5, edgecolors='k', label='BE(s)')
        y_r = max(abs(PnL.min()), abs(PnL.max())) if len(PnL)>0 and PnL.min() is not None and PnL.max() is not None else 1
        offset = y_r*0.05 if y_r != 0 else 0.1
        for be in valid_bes: ax.text(be, offset, f'BE: {format_currency(be, currency)}', c='darkred', ha='center', va='bottom', fontsize=9, weight='bold')
    p_lbl = f'Max Profit: {format_currency(max_p, currency)}' if np.isfinite(max_p) else 'Max Profit: Unlimited'
    l_lbl = f'Max Loss: {format_currency(max_l, currency)}' if np.isfinite(max_l) else 'Max Loss: Unlimited'
    if np.isfinite(max_p): ax.axhline(max_p, c='g', ls=':', lw=1.5, label=p_lbl)
    elif len(S_T_range)>0 and len(PnL)>0 and pd.notna(PnL[-1]): ax.text(S_T_range[-1], PnL[-1], ' Unlim Profit ->', c='darkgreen', ha='right', va='center', weight='bold', bbox=dict(fc='w', alpha=0.5, pad=0.1))
    if np.isfinite(max_l): ax.axhline(max_l, c='r', ls=':', lw=1.5, label=l_lbl)
    elif len(S_T_range)>0 and len(PnL)>0 and pd.notna(PnL[0]): ax.text(S_T_range[0], PnL[0], '<- Unlim Loss ', c='darkred', ha='left', va='center', weight='bold', bbox=dict(fc='w', alpha=0.5, pad=0.1))
    ax.set_title(f'{strategy_name} Payoff', fontsize=15, weight='bold'); ax.set_xlabel(f'Price@Exp ({currency})', fontsize=11); ax.set_ylabel(f'Profit/Loss ({currency})', fontsize=11); ax.grid(True, ls=':', c='grey')
    y_min, y_max = (np.nanmin(PnL) if len(PnL)>0 else -1), (np.nanmax(PnL) if len(PnL)>0 else 1)
    if np.isfinite(max_l): y_min = min(y_min, max_l);
    if np.isfinite(max_p): y_max = max(y_max, max_p);
    pad = (y_max-y_min)*0.1 if (y_max != y_min) else 1.0; ax.set_ylim(y_min-pad, y_max+pad)
    ax.legend(fontsize=9); plt.tight_layout();
    try: plt.show()
    except Exception as e: print(f"\nPlot display Err: {e}.")