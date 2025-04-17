# bsm_formulas.py
import numpy as np
from scipy.stats import norm
from utils import safe_float

def black_scholes_merton(S, K, T, r, q, sigma, option_type="call"):
    """ BSM Option Price including continuous dividend yield q. """
    S, K, T, r, q, sigma = map(safe_float, [S, K, T, r, q, sigma])
    if any(np.isnan(x) for x in [S, K, T, r, q, sigma]): return np.nan
    if S <= 0 or K <= 0: return np.nan # Prices must be positive

    # --- Explicit Handling for T=0 (Expiration) ---
    if T <= 1e-9: # Use small threshold instead of exact zero for float comparison
        price = 0.0
        if option_type.lower() == "call": price = max(0.0, S - K)
        elif option_type.lower() == "put": price = max(0.0, K - S)
        else: return np.nan # Invalid type
        # print(f"Debug BSM T=0: S={S}, K={K}, type={option_type}, Intrinsic={price}") # Debug Line
        return price

    # --- Handling for T > 0 ---
    if sigma <= 0: sigma = 1e-6 # Minimal volatility if input is zero/negative

    try:
        sqrt_T = np.sqrt(T)
        denom = sigma * sqrt_T
        # Check for potential division by zero if sigma or T is effectively zero
        if abs(denom) < 1e-9:
            # If no volatility or time, value is discounted intrinsic value
            if option_type.lower() == "call": return max(0.0, S * np.exp(-q*T) - K * np.exp(-r*T))
            elif option_type.lower() == "put": return max(0.0, K * np.exp(-r*T) - S * np.exp(-q*T))
            else: return np.nan

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / denom
        d2 = d1 - denom

        if option_type.lower() == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            return np.nan
        return max(0.0, price) # Ensure non-negative price

    except (OverflowError, ValueError): return np.nan # Catch math errors
    except Exception: return np.nan # Catch any other unexpected error

def calculate_option_greeks(S, K, T, r, q, sigma, option_type="call"):
    """ Calculate option Greeks including dividend yield q. """
    greeks = { "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan }
    S, K, T, r, q, sigma = map(safe_float, [S, K, T, r, q, sigma])
    if any(np.isnan(x) for x in [S, K, T, r, q, sigma]): return greeks
    if S <= 0 or K <= 0: return greeks

    opt_type_lower = option_type.lower()

    # --- Explicit Handling for T=0 (Expiration) ---
    if T <= 1e-9:
        delta = np.nan
        if opt_type_lower == 'call': delta = 1.0 if S > K else (0.5 if abs(S - K) < 1e-9 else 0.0)
        elif opt_type_lower == 'put': delta = -1.0 if S < K else (-0.5 if abs(S - K) < 1e-9 else 0.0)
        else: return greeks # Invalid type
        # print(f"Debug Greeks T=0: S={S}, K={K}, type={opt_type_lower}, Delta={delta}") # Debug Line
        return { "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0 }

    # --- Handling for T > 0 ---
    if sigma <= 0: sigma = 1e-6 # Minimal volatility

    try:
        sqrt_T = np.sqrt(T)
        denom = sigma * sqrt_T
        if abs(denom) < 1e-9: return {k: 0.0 for k in greeks} # Zero greeks if no time/vol

        exp_qT, exp_rT = np.exp(-q * T), np.exp(-r * T)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / denom
        d2 = d1 - denom
        pdf_d1 = norm.pdf(d1)
        cdf_d1, cdf_d2 = norm.cdf(d1), norm.cdf(d2)
        cdf_neg_d1, cdf_neg_d2 = norm.cdf(-d1), norm.cdf(-d2)

        gamma_denom = S * denom
        greeks["gamma"] = (exp_qT * pdf_d1 / gamma_denom) if gamma_denom != 0 else 0
        greeks["vega"] = (S * exp_qT * sqrt_T * pdf_d1) / 100 # Per 1% change

        # Theta (per day)
        theta_term1 = - (S * exp_qT * pdf_d1 * sigma) / (2 * sqrt_T) if sqrt_T > 1e-9 else 0

        if opt_type_lower == "call":
            greeks["delta"] = exp_qT * cdf_d1
            theta_term2, theta_term3 = -r * K * exp_rT * cdf_d2, +q * S * exp_qT * cdf_d1
            greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
            greeks["rho"] = (K * T * exp_rT * cdf_d2) / 100 # Per 1% change in r
        elif opt_type_lower == "put":
            greeks["delta"] = exp_qT * (cdf_d1 - 1)
            theta_term2, theta_term3 = +r * K * exp_rT * cdf_neg_d2, -q * S * exp_qT * cdf_neg_d1
            greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365
            greeks["rho"] = (-K * T * exp_rT * cdf_neg_d2) / 100 # Per 1% change in r
        else:
            return {k: np.nan for k in greeks}

        return greeks

    except (ZeroDivisionError, OverflowError, ValueError): return {k: np.nan for k in greeks}
    except Exception: return {k: np.nan for k in greeks}

def calculate_implied_volatility(S, K, T, r, q, market_price, option_type="call",
                                  precision=0.0001, max_iterations=100):
    """ Calculate implied volatility using bisection. """
    S, K, T, r, q, market_price = map(safe_float, [S, K, T, r, q, market_price])
    if any(np.isnan(x) for x in [S, K, T, r, q, market_price]): return np.nan
    opt_type = option_type.lower()
    if market_price <= 0 or T <= 1e-9 or S <= 0 or K <= 0: return np.nan # Cannot calculate IV at T=0 or for invalid inputs

    try: # Calculate intrinsic value robustly
        intrinsic = black_scholes_merton(S, K, T, r, q, 1e-6, opt_type) # Use near-zero vol
        if np.isnan(intrinsic): return np.nan # Failed to calculate intrinsic
    except Exception: return np.nan

    if market_price < intrinsic - precision: return np.nan # Price below intrinsic

    # Bisection method
    vol_low, vol_high = 1e-5, 5.0
    for _ in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2
        price_mid = black_scholes_merton(S, K, T, r, q, vol_mid, opt_type)

        if np.isnan(price_mid): # Handle BSM failure during iteration
             # Try reducing vol range if price seems too high/low
             if market_price > black_scholes_merton(S, K, T, r, q, vol_low, opt_type):
                 vol_low = vol_mid # Need higher vol, narrow from below
             else:
                 vol_high = vol_mid # Need lower vol, narrow from above
             continue # Skip convergence check this iteration

        diff = price_mid - market_price
        if abs(diff) < precision: return vol_mid # Converged on price

        if diff > 0: vol_high = vol_mid
        else: vol_low = vol_mid

        if abs(vol_high - vol_low) < precision: break # Converged on volatility

    # Return midpoint after loop if price is reasonably close
    final_vol = (vol_low + vol_high) / 2
    final_price = black_scholes_merton(S, K, T, r, q, final_vol, opt_type)
    return final_vol if pd.notna(final_price) and abs(final_price - market_price) < precision * 10 else np.nan