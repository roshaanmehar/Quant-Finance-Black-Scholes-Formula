# config_manager.py
import json
import os

DEFAULT_CONFIG = {
    'volatility_days': 252,
    'default_risk_free_rate': 0.04,
    'show_greeks_in_chain': True,
    'max_strikes_chain': 20,
    'iv_precision': 0.0001,
    'iv_max_iterations': 100,
    'strategy_price_range': 0.3,
    'debug_mode': False
}

CONFIG_PATH = 'options_config.json'
FAVORITES_PATH = 'favorite_tickers.json'

def load_config():
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config_data = json.load(f)
                config.update(config_data) # Overwrite defaults with loaded values
            print("Configuration loaded.")
        else:
            print("No config file found, using defaults.")
    except json.JSONDecodeError:
        print(f"Error reading config '{CONFIG_PATH}'. Using defaults.")
        config = DEFAULT_CONFIG.copy() # Reset to default on error
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        config = DEFAULT_CONFIG.copy() # Reset to default on error
    return config

def save_config(config_data):
    """Save configuration to file."""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)
        print("Configuration saved.")
    except Exception as e:
        print(f"Error saving config: {e}")

def load_favorites():
    """Load favorite tickers from file."""
    try:
        if os.path.exists(FAVORITES_PATH):
            with open(FAVORITES_PATH, 'r') as f:
                favorites = json.load(f)
            if isinstance(favorites, list):
                print(f"Loaded {len(favorites)} favorites.")
                return sorted(list(set(favorites))) # Ensure unique and sorted
            else:
                print("Error: Favorites file format incorrect. Resetting.")
                return []
        else:
            print("No favorites file found.")
            return []
    except json.JSONDecodeError:
        print(f"Error reading favorites '{FAVORITES_PATH}'. Resetting.")
        return []
    except Exception as e:
        print(f"Error loading favorites: {e}. Resetting.")
        return []

def save_favorites(favorites_list):
    """Save favorite tickers to file."""
    try:
        # Ensure list contains unique strings before saving
        valid_favorites = sorted(list(set(fav for fav in favorites_list if isinstance(fav, str))))
        with open(FAVORITES_PATH, 'w') as f:
            json.dump(valid_favorites, f, indent=4)
        print("Favorites saved.")
    except Exception as e:
        print(f"Error saving favorites: {e}")