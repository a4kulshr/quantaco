import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.linear_model import RidgeCV
import seaborn as sns
import time
from datetime import datetime, timedelta
import logging
import os
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_data(ticker, start_date, end_date):
    """Get data from cache if available and not expired"""
    cache_file = CACHE_DIR / f"{ticker}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        # Check if cache is less than 1 day old
        if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 1:
            logging.info(f"Loading cached data for {ticker}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    return None

def save_to_cache(data, ticker, start_date, end_date):
    """Save data to cache"""
    cache_file = CACHE_DIR / f"{ticker}_{start_date}_{end_date}.csv"
    data.to_csv(cache_file)
    logging.info(f"Saved data to cache for {ticker}")

def get_stock_data(ticker, start_date, end_date, max_retries=5, delay=5):
    """
    Fetch stock data from yfinance with retry logic, rate limit handling, and caching
    """
    # Try to get data from cache first
    cached_data = get_cached_data(ticker, start_date, end_date)
    if cached_data is not None:
        return cached_data['Close'].pct_change().dropna()

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1} to fetch data for {ticker}")
            stock = yf.Ticker(ticker)
            
            # Split the date range into smaller chunks to avoid rate limits
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            chunk_size = timedelta(days=30)  # 1 month chunks
            
            all_data = []
            current_start = start
            
            while current_start < end:
                current_end = min(current_start + chunk_size, end)
                logging.info(f"Fetching data from {current_start.date()} to {current_end.date()}")
                
                # Add longer delay between chunks
                time.sleep(delay)
                
                df_chunk = stock.history(start=current_start.strftime('%Y-%m-%d'),
                                       end=current_end.strftime('%Y-%m-%d'),
                                       progress=False)  # Disable progress bar
                
                if not df_chunk.empty:
                    all_data.append(df_chunk)
                
                current_start = current_end
                time.sleep(delay)  # Add delay between requests
            
            if not all_data:
                raise ValueError(f"No data found for {ticker}")
            
            # Combine all chunks
            df = pd.concat(all_data)
            df = df[~df.index.duplicated(keep='first')]  # Remove any duplicate dates
            
            # Save to cache
            save_to_cache(df, ticker, start_date, end_date)
            
            return df['Close'].pct_change().dropna()
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                logging.error(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                raise
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(delay * (attempt + 1))  # Exponential backoff

def calculate_weights_and_sharpe(returns, sigma=0.04):
    """
    Calculate portfolio weights and Sharpe ratio using the methodology from RSM338_A4.py
    """
    # Calculate mean returns and covariance matrix
    mu = returns.mean()
    S = returns.cov()
    
    # Ensure covariance matrix is symmetric
    S = (S + S.T) / 2
    
    # Calculate inverse of covariance matrix
    inv_S = inv(S)
    
    # Calculate maximum Sharpe ratio weights
    w = inv_S.dot(mu)
    theta = np.dot(mu, w)  # sample maximum squared Sharpe ratio
    w_optimal = sigma/np.sqrt(theta) * w
    
    # Calculate Sharpe ratio (annualized)
    sharpe_ratio = np.sqrt(252) * np.dot(w_optimal, mu) / np.sqrt(np.dot(w_optimal, S.dot(w_optimal)))
    
    return w_optimal, sharpe_ratio

def plot_weights(weights, ticker):
    """Plot portfolio weights"""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights, color='blue', alpha=0.6)
    plt.title(f'Portfolio Weights for {ticker}')
    plt.xlabel('Asset Index')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Parameters - Using a shorter time period
    ticker = 'AMT'
    start_date = '2023-01-01'  # Changed to last year only
    end_date = '2024-01-01'
    sigma = 0.04  # Risk constraint
    
    try:
        # Get stock data
        logging.info(f"Fetching data for {ticker}...")
        returns = get_stock_data(ticker, start_date, end_date)
        
        # Calculate weights and Sharpe ratio
        logging.info("Calculating optimal weights and Sharpe ratio...")
        weights, sharpe = calculate_weights_and_sharpe(returns, sigma)
        
        # Print results
        print(f"\nResults for {ticker}:")
        print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
        print("\nPortfolio Weights:")
        for i, w in enumerate(weights):
            print(f"Asset {i+1}: {w:.4f}")
        
        # Plot weights
        plot_weights(weights, ticker)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
