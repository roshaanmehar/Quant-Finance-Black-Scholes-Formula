# options_formulas.py
# Standalone functions for Black-Scholes-Merton, Greeks, and IV

import numpy as np
from scipy.stats import norm
from utils import safe_float # Import from utils module

def black_scholes_merton(S, K, T, r, q, sigma, option_type="call"):
    """
    Calculate BSM Option Price including continuous dividend yield q.
    Handles T=0 expiration correctly (returns intrinsic value).
    Ensures inputs are numeric. Returns np.nan on error or invalid inputs.
    """
    S, K, T, r, q, sigma = map(safe_float, [S, K, T, r, q, sigma])
    if any(np.isnan(x) for x in [S, K, T, r, q, sigma]): return np.nan
    if T < 0: T = 0 # Treat past dates as T=0
    if sigma <= 0: sigma = 1e-9 # Use extremely small vol instead of zero
    if S <= 0 or K <= 0: return np.nan # Prices must be positive

    # --- Handle T=0 (Expiration) ---
    if T == 0:
        if option_type.lower() == "call": return max(0.0, S - K)
        elif option_type.lower() == "put": return max(0.0, K - S)
        else: return np.nan

    # --- Standard BSM Calculation (T > 0) ---
    try:
        sqrt_T = np.sqrt(T)
        denom = sigma * sqrt_T
        # Check for division by zero (shouldn't happen with sigma check above, but safe)
        if denom == 0: return np.nan

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / denom
        d2 = d1 - denom

        if option_type.lower() == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            return np.nan
        # Ensure price is not negative due to potential floating point inaccuracies
        return max(0.0, price)

    except (OverflowError, ValueError): return np.nan # Catch math errors
    except Exception: return np.nan # Catch any other unexpected errors

def calculate_option_greeks(S, K, T, r, q, sigma, option_type="call"):
    """
    Calculate option Greeks including dividend yield q.
    Handles T=0 expiration correctly (defines Delta, others zero/NaN).
    Ensures inputs are numeric. Returns dict with np.nan on errors.
    """
    greeks = { "delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan }
    S, K, T, r, q, sigma = map(safe_float, [S, K, T, r, q, sigma])
    if any(np.isnan(x) for x in [S, K, T, r, q, sigma]): return greeks

    # --- Handle T=0 (Expiration) ---
    if T <= 0:
        opt_type = option_type.lower()
        delta = np.nan
        # Delta at expiration is 0, 0.5, or 1 (or -1 for puts)
        if opt_type == 'call':
            if S > K: delta = 1.0
            elif S == K: delta = 0.5
            else: delta = 0.0
        elif opt_type == 'put':
            if S < K: delta = -1.0
            elif S == K: delta = -0.5
            else: delta = 0.0
        # Other greeks are generally considered 0 at expiration
        return { "delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0 }

    # --- Handle other invalid inputs (before T=0 check) ---
    if sigma <= 0 or S <= 0 or K <= 0:
        # If T>0 but other inputs invalid, return all NaNs
        return greeks

    # --- Standard Greeks Calculation (T > 0, sigma > 0) ---
    opt_type = option_type.lower()
    try:
        sqrt_T = np.sqrt(T)
        denom = sigma * sqrt_T
        # Should not happen due to checks above, but safety
        if denom == 0: return {k: 0.0 for k in greeks}

        exp_qT, exp_rT = np.exp(-q * T), np.exp(-r * T)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / denom
        d2 = d1 - denom
        pdf_d1 = norm.pdf(d1)
        cdf_d1, cdf_d2 = norm.cdf(d1), norm.cdf(d2)
        cdf_neg_d1, cdf_neg_d2 = norm.cdf(-d1), norm.cdf(-d2)

        # Gamma (Ensure denominator is non-zero)
        gamma_denom = S * denom
        greeks["gamma"] = (exp_qT * pdf_d1 / gamma_denom) if gamma_denom != 0 else 0.0

        # Vega (per 1%)
        greeks["vega"] = (S * exp_qT * sqrt_T * pdf_d1) / 100.0

        # Theta (per day)
        theta_term1 = - (S * exp_qT * pdf_d1 * sigma) / (2 * sqrt_T)

        if opt_type == "call":
            greeks["delta"] = exp_qT * cdf_d1
            theta_term2 = -r * K * exp_rT * cdf_d2
            theta_term3 = +q * S * exp_qT * cdf_d1
            greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365.0
            # Rho (per 1%)
            greeks["rho"] = (K * T * exp_rT * cdf_d2) / 100.0
        elif opt_type == "put":
            greeks["delta"] = exp_qT * (cdf_d1 - 1)
            theta_term2 = +r * K * exp_rT * cdf_neg_d2
            theta_term3 = -q * S * exp_qT * cdf_neg_d1
            greeks["theta"] = (theta_term1 + theta_term2 + theta_term3) / 365.0
            # Rho (per 1%)
            greeks["rho"] = (-K * T * exp_rT * cdf_neg_d2) / 100.0
        else:
            # Invalid option type, return NaNs
            return {k: np.nan for k in greeks}

        # Final check for NaN in results (could happen with extreme inputs)
        for k, v in greeks.items():
            if np.isnan(v):
                greeks[k] = np.nan # Ensure consistency

        return greeks

    except (ZeroDivisionError, OverflowError, ValueError): return {k: np.nan for k in greeks}
    except Exception: return {k: np.nan for k in greeks}

def calculate_implied_volatility(S, K, T, r, q, market_price, option_type="call",
                                 precision=0.0001, max_iterations=100):
    """
    Calculate implied volatility using bisection. Ensures numeric inputs.
    Returns np.nan on errors or if convergence fails.
    """
    S, K, T, r, q, market_price = map(safe_float, [S, K, T, r, q, market_price])
    if any(np.isnan(x) for x in [S, K, T, r, q, market_price]): return np.nan

    opt_type = option_type.lower()

    # --- Basic Input Validation ---
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0: return np.nan

    # --- Check vs Intrinsic Value ---
    try:
        intrinsic_value = 0.0
        if opt_type == "call": intrinsic_value = max(0.0, S * np.exp(-q*T) - K * np.exp(-r * T))
        elif opt_type == "put": intrinsic_value = max(0.0, K * np.exp(-r * T) - S * np.exp(-q*T))
        else: return np.nan # Invalid type
    except OverflowError: return np.nan # Cannot calculate intrinsic

    # If price is below intrinsic (allowing for tiny precision error), IV is undefined/meaningless
    if market_price < intrinsic_value - precision: return np.nan

    # --- Bisection Method ---
    vol_low, vol_high = 1e-5, 5.0 # Volatility bounds (0.001% to 500%)
    vol_mid = 0.0 # Initialize

    for _ in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2.0
        # Prevent vol_mid from becoming exactly zero if bounds are tiny
        if vol_mid < 1e-9: vol_mid = 1e-9

        price_mid = black_scholes_merton(S, K, T, r, q, vol_mid, opt_type)

        # Handle BSM failure during iteration
        if np.isnan(price_mid):
            # Try adjusting bounds slightly - if BSM fails repeatedly, IV calc likely impossible
            # This indicates potential instability, returning NaN is safest
            return np.nan

        diff = price_mid - market_price

        # Check for convergence
        if abs(diff) < precision: return vol_mid

        # Narrow the search interval
        if diff > 0: vol_high = vol_mid
        else: vol_low = vol_mid

        # Check if bounds are too close (meaning convergence failed within range)
        if abs(vol_high - vol_low) < precision: break

    # --- After loop: Check if final estimate is reasonable ---
    final_vol = (vol_low + vol_high) / 2.0
    # Ensure final vol is positive
    if final_vol <= 0: return np.nan

    final_price = black_scholes_merton(S, K, T, r, q, final_vol, opt_type)

    # Return vol only if the final price is close enough to market price
    if pd.notna(final_price) and abs(final_price - market_price) < precision * 10: # Looser check
         return final_vol
    else:
         # Failed to converge reliably
         return np.nan

# You could add other financial formula functions here, e.g., different option models