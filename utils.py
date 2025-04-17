# utils.py
# Utility functions for the Options Analyzer

import numpy as np
import os

def safe_float(value, default=np.nan):
    """Safely converts a value to float, returning default on error."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def format_currency(value, currency='USD'):
    """Formats a numeric value as currency."""
    numeric_value = safe_float(value, default=None)
    if numeric_value is None:
        return "N/A"
    try:
        symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
        symbol = symbols.get(currency, '')
        return f"{symbol}{numeric_value:,.2f}"
    except (TypeError, ValueError):
         return "N/A"

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Add any other general utility functions here if needed