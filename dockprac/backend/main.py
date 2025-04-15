from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import math

app = FastAPI(title="Option Pricing API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptionInput(BaseModel):
    ticker: str
    strike_price: float
    time_to_expiry: float  # in years
    risk_free_rate: float = 0.05  # default 5%
    volatility: float = None  # if None, will be calculated from historical data

@app.get("/")
def read_root():
    return {"message": "Welcome to the Option Pricing API"}

@app.get("/get_stock_price/{ticker}")
async def get_stock_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        # Calculate historical volatility (30-day)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Calculate daily returns
        hist_data['return'] = hist_data['Close'].pct_change()
        # Calculate annualized volatility
        volatility = hist_data['return'].std() * np.sqrt(252)  # 252 trading days in a year
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "volatility": volatility,
            "company_name": info.get('shortName', ticker)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.post("/calculate_option")
async def calculate_option(option_input: OptionInput):
    try:
        # If volatility is not provided, fetch it
        if option_input.volatility is None:
            stock_data = await get_stock_price(option_input.ticker)
            volatility = stock_data["volatility"]
            stock_price = stock_data["current_price"]
        else:
            volatility = option_input.volatility
            # Fetch current stock price
            stock_data = await get_stock_price(option_input.ticker)
            stock_price = stock_data["current_price"]
        
        # TODO: Implement Black-Scholes formula here
        # This is where you'll implement the formula
        
        # Placeholder for Black-Scholes calculation
        # You will replace this with your implementation
        d1 = (np.log(stock_price / option_input.strike_price) + 
              (option_input.risk_free_rate + 0.5 * volatility**2) * option_input.time_to_expiry) / \
             (volatility * np.sqrt(option_input.time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(option_input.time_to_expiry)
        
        # Calculate call option price
        call_price = stock_price * norm.cdf(d1) - \
                    option_input.strike_price * np.exp(-option_input.risk_free_rate * option_input.time_to_expiry) * norm.cdf(d2)
        
        # Calculate put option price
        put_price = option_input.strike_price * np.exp(-option_input.risk_free_rate * option_input.time_to_expiry) * norm.cdf(-d2) - \
                   stock_price * norm.cdf(-d1)
        
        return {
            "ticker": option_input.ticker,
            "stock_price": stock_price,
            "strike_price": option_input.strike_price,
            "time_to_expiry": option_input.time_to_expiry,
            "risk_free_rate": option_input.risk_free_rate,
            "volatility": volatility,
            "call_price": call_price,
            "put_price": put_price,
            "d1": d1,
            "d2": d2
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating option price: {str(e)}")

@app.get("/historical_data/{ticker}")
async def get_historical_data(ticker: str, days: int = 30):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for ticker {ticker}")
        
        # Convert to list of dictionaries for JSON response
        result = []
        for date, row in hist_data.iterrows():
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"]
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

