# Options Analyzer Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A comprehensive quantitative finance toolkit for options pricing, analysis, and strategy evaluation using the Black-Scholes-Merton model.**

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [API Reference](#api-reference) | [Architecture](#architecture)

</div>

---

## Overview

Options Analyzer Pro is a professional-grade options analysis platform that provides real-time stock data fetching, theoretical option pricing using the Black-Scholes-Merton model with dividend adjustments, Greeks calculation, implied volatility computation, and comprehensive strategy analysis with payoff visualizations.

The project includes multiple interfaces:
- **Console Application** - Interactive CLI for quick analysis
- **Streamlit Web App** - Full-featured web dashboard
- **FastAPI Backend** - RESTful API for integration
- **Next.js Frontend** - Modern React-based UI (Docker deployment)

---

## Features

### Core Pricing Engine

| Feature | Description |
|---------|-------------|
| **Black-Scholes-Merton** | Full BSM implementation with continuous dividend yield adjustment |
| **Greeks Calculation** | Delta, Gamma, Theta, Vega, and Rho for both calls and puts |
| **Implied Volatility** | Bisection method IV solver with convergence optimization |
| **Real-time Data** | Live stock prices and historical volatility via yfinance |
| **Risk-Free Rate** | Automatic 10-Year Treasury yield fetching |

### Options Chain Analysis

- Fetch complete options chains for any expiration date
- Side-by-side comparison of market prices vs. BSM theoretical prices
- Calculated and Yahoo Finance implied volatility comparison
- Volume and open interest analysis
- Interactive visualization with IV smile/skew charts

### Strategy Analysis

Analyze popular options strategies with automatic calculation of:
- **Breakeven points**
- **Maximum profit/loss**
- **Payoff diagrams**

| Strategy | Description |
|----------|-------------|
| Covered Call | Long stock + Short call |
| Protective Put | Long stock + Long put |
| Bull Call Spread | Long lower strike call + Short higher strike call |
| Bear Put Spread | Long higher strike put + Short lower strike put |
| Long Straddle | Long ATM call + Long ATM put |
| Long Strangle | Long OTM call + Long OTM put |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 18+ (for Next.js frontend)
- Docker & Docker Compose (optional, for containerized deployment)

### Quick Start (Python)

```bash
# Clone the repository
git clone https://github.com/yourusername/options-analyzer-pro.git
cd options-analyzer-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas yfinance matplotlib scipy tabulate streamlit
```

### Docker Deployment

```bash
cd dockprac
docker-compose up --build
```

This starts:
- **Backend API** at `http://localhost:8000`
- **Frontend** at `http://localhost:3000`

---

## Usage

### Console Application

```bash
python main_console.py
```

Navigate the interactive menu:
```
+===================================+
|     Options Analyzer Menu         |
+===================================+
  1. Fetch Data
  2. Simple Price
  3. Options Chain
  4. Analyze Strategy
  5. Manage Favs
  6. Settings
  0. Exit
-------------------------------------
```

### Streamlit Web App

```bash
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

### Python Module Usage

```python
from option import OptionsAnalyzer

# Initialize analyzer
analyzer = OptionsAnalyzer()

# Fetch stock data
stock_data = analyzer.get_stock_data("AAPL")

# Calculate option price
price = analyzer.black_scholes_merton(
    S=150.0,      # Stock price
    K=155.0,      # Strike price
    T=0.25,       # Time to expiration (years)
    r=0.05,       # Risk-free rate
    q=0.006,      # Dividend yield
    sigma=0.25,   # Volatility
    option_type="call"
)

# Calculate Greeks
greeks = analyzer.calculate_option_greeks(
    S=150.0, K=155.0, T=0.25, r=0.05, q=0.006, sigma=0.25, option_type="call"
)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f} per day")
print(f"Vega: {greeks['vega']:.4f} per 1%")
print(f"Rho: {greeks['rho']:.4f} per 1%")
```

---

## API Reference

### FastAPI Endpoints

#### Get Stock Price
```http
GET /get_stock_price/{ticker}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "current_price": 178.50,
  "volatility": 0.2456,
  "company_name": "Apple Inc."
}
```

#### Calculate Option Price
```http
POST /calculate_option
```

**Request Body:**
```json
{
  "ticker": "AAPL",
  "strike_price": 180.0,
  "time_to_expiry": 0.25,
  "risk_free_rate": 0.05,
  "volatility": null
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "stock_price": 178.50,
  "strike_price": 180.0,
  "time_to_expiry": 0.25,
  "risk_free_rate": 0.05,
  "volatility": 0.2456,
  "call_price": 8.45,
  "put_price": 7.67,
  "d1": 0.1234,
  "d2": 0.0011
}
```

#### Get Historical Data
```http
GET /historical_data/{ticker}?days=30
```

---

## Architecture

```
options-analyzer-pro/
├── Core Engine
│   ├── option.py              # Main OptionsAnalyzer class (monolithic)
│   ├── options_formulas.py    # BSM, Greeks, IV calculations
│   ├── options_analyzer_core.py # Modular analyzer core
│   ├── strategies.py          # Strategy payoff calculations
│   ├── data_fetcher.py        # Stock data & rate fetching
│   ├── plotting.py            # Visualization functions
│   └── utils.py               # Utility functions
│
├── Interfaces
│   ├── main_console.py        # Console application
│   ├── app.py                 # Streamlit web app
│   └── config_manager.py      # Configuration management
│
├── Docker Deployment (dockprac/)
│   ├── backend/
│   │   ├── main.py            # FastAPI application
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── app/               # Next.js app
│   │   └── Dockerfile
│   └── docker-compose.yml
│
└── Configuration
    ├── options_config.json    # User settings
    └── favorite_tickers.json  # Saved tickers
```

### Key Components

| Module | Purpose |
|--------|---------|
| `options_formulas.py` | Pure mathematical functions for BSM pricing, Greeks, and IV |
| `options_analyzer_core.py` | Data management, caching, and orchestration |
| `data_fetcher.py` | yfinance integration for market data |
| `strategies.py` | Multi-leg strategy payoff calculations |
| `plotting.py` | Matplotlib visualizations |

---

## Mathematical Background

### Black-Scholes-Merton Formula

The BSM model with continuous dividend yield:

**Call Option:**
$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

**Put Option:**
$$P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

Where:
$$d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

| Symbol | Description |
|--------|-------------|
| $S_0$ | Current stock price |
| $K$ | Strike price |
| $T$ | Time to expiration (years) |
| $r$ | Risk-free interest rate |
| $q$ | Continuous dividend yield |
| $\sigma$ | Volatility |
| $N(x)$ | Standard normal CDF |

### Greeks

| Greek | Call Formula | Interpretation |
|-------|--------------|----------------|
| **Delta** | $e^{-qT} N(d_1)$ | Price sensitivity to underlying |
| **Gamma** | $\frac{e^{-qT} N'(d_1)}{S_0 \sigma \sqrt{T}}$ | Delta sensitivity |
| **Theta** | Complex | Time decay per day |
| **Vega** | $S_0 e^{-qT} \sqrt{T} N'(d_1)$ | Volatility sensitivity |
| **Rho** | $KT e^{-rT} N(d_2)$ | Interest rate sensitivity |

---

## Configuration

Create `options_config.json` to customize behavior:

```json
{
  "volatility_days": 252,
  "default_risk_free_rate": 0.04,
  "show_greeks_in_chain": true,
  "max_strikes_chain": 20,
  "iv_precision": 0.0001,
  "iv_max_iterations": 100,
  "strategy_price_range": 0.3,
  "debug_mode": false
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `volatility_days` | Trading days for annualization | 252 |
| `default_risk_free_rate` | Fallback rate if Treasury fetch fails | 0.04 |
| `show_greeks_in_chain` | Include Greeks in chain output | true |
| `max_strikes_chain` | Maximum strikes to display | 20 |
| `iv_precision` | IV calculation convergence threshold | 0.0001 |
| `iv_max_iterations` | Maximum IV solver iterations | 100 |
| `strategy_price_range` | Price range for payoff diagrams (% of S0) | 0.3 |
| `debug_mode` | Enable detailed error output | false |

---

## Screenshots
![](https://github.com/roshaanmehar/Quant-Finance-Black-Scholes-Formula/blob/main/Screenshot%202026-01-28%20105449.png)
![](https://github.com/roshaanmehar/Quant-Finance-Black-Scholes-Formula/blob/main/Screenshot%202026-01-28%20105502.png)
![](https://github.com/roshaanmehar/Quant-Finance-Black-Scholes-Formula/blob/main/Screenshot%202026-01-28%20105518.png)
![](https://github.com/roshaanmehar/Quant-Finance-Black-Scholes-Formula/blob/main/Screenshot%202026-01-28%20105602.png)
![}(https://github.com/roshaanmehar/Quant-Finance-Black-Scholes-Formula/blob/main/Screenshot%202026-01-28%20105615.png)

## Dependencies

### Python
```
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.2.0
matplotlib>=3.4.0
scipy>=1.7.0
tabulate>=0.8.9
streamlit>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
```

### Node.js (Frontend)
```
next>=15.0.0
react>=19.0.0
tailwindcss>=4.0.0
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data access
- [Streamlit](https://streamlit.io/) for the web dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- Black, Scholes, and Merton for the foundational pricing model

---

<div align="center">

**Built with precision for quantitative finance professionals**

</div>
