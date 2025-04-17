# strategies.py
import numpy as np
from utils import safe_float

def calculate_strategy_payoff(S_T, strategy_legs):
    """Calculates the Profit/Loss of a multi-leg strategy at expiration price S_T."""
    S_T_num = safe_float(S_T)
    if np.isnan(S_T_num): return np.nan

    total_terminal_value = 0
    net_initial_cost = 0

    for leg in strategy_legs:
        price = safe_float(leg.get('price'), 0.0) # Initial cost/credit of the leg
        K = safe_float(leg.get('K')) # Strike (NaN if not an option or invalid)
        direction = leg.get('dir', 'long')
        leg_type = leg.get('type')

        # Accumulate net initial cost
        if direction == 'long': net_initial_cost += price
        else: net_initial_cost -= price # Subtract credit

        # Calculate terminal value of this leg
        leg_terminal_value = 0
        if leg_type == 'stock':
            leg_terminal_value = S_T_num
        elif leg_type == 'call' and not np.isnan(K):
            leg_terminal_value = max(0.0, S_T_num - K)
        elif leg_type == 'put' and not np.isnan(K):
            leg_terminal_value = max(0.0, K - S_T_num)

        # If short, the terminal value contribution is negative
        if direction == 'short':
            leg_terminal_value *= -1

        total_terminal_value += leg_terminal_value

    # Profit/Loss = Total Terminal Value - Net Initial Cost
    profit_loss = total_terminal_value - net_initial_cost
    return profit_loss

# Example strategy definition (could be expanded)
# def define_covered_call(stock_price, call_strike, call_premium):
#     legs = [{'type': 'stock', 'dir': 'long', 'price': stock_price},
#             {'type': 'call', 'dir': 'short', 'K': call_strike, 'price': call_premium}]
#     cost = stock_price - call_premium
#     be = cost
#     max_p = call_strike - cost
#     max_l = -cost
#     return legs, be, max_p, max_l