import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import seaborn as sns
import yfinance as yf

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_weights_and_sharpe(returns, sigma=0.04):
    mu = returns.mean()
    S = returns.cov()
    inverse_S = np.linalg.inv(S)
    S = (S + S.T) / 2
    inv_S = inv(S)
    w = inv_S.dot(mu) # mu *  transpose of mu
    theta = np.dot(mu, w)
# Scale weights to target volatility
    portfolio_vol = np.sqrt(np.dot(w.T, S.dot(w)))
    w_optimal = (sigma / portfolio_vol) * w

    # Normalize weights to sum to 1
    if (w_optimal < 0).any():
        w_optimal = np.maximum(w_optimal, 0)
    if w_optimal.sum() > 0:
        w_optimal /= w_optimal.sum()
    else:
        w_optimal = np.ones_like(w_optimal) / len(w_optimal)
    
    # Calculate annualized Sharpe
    portfolio_return = np.dot(w_optimal, mu)
    portfolio_vol = np.sqrt(np.dot(w_optimal.T, S.dot(w_optimal)))
    sharpe_ratio = np.sqrt(252) * portfolio_return / portfolio_vol
    return w_optimal, sharpe_ratio

def plot_weights(weights, tickers):
    plt.figure(figsize=(8, 5))
    plt.bar(tickers, weights, color='blue', alpha=0.6)
    plt.title('Portfolio Weights')
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Load data from CSV
    df = pd.read_csv('AMT CCI Data.csv', parse_dates=['Date'])
    df = df[['Date', 'Close_x']].dropna()
    df = df.rename(columns={'Close_x': 'AMT'})
    df.set_index('Date', inplace=True)
    #set ^GSPC as risk free second asset
    risk_free = yf.download('^IRX', period ='5y', interval='1d')
    df = pd.concat([df, risk_free[['Close']].rename(columns={'Close': 'RiskFree'})], axis=1, join='inner')
    df.rename(columns={'Close': 'RiskFree'}, inplace=True)
    df = df.dropna()

    # Calculate daily returns
    returns = df.pct_change().dropna()

    # Calculate weights and Sharpe ratio
    weights, sharpe = calculate_weights_and_sharpe(returns, sigma=0.04)

    # Print results
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
    print("Portfolio Weights:")
    for ticker, w in zip(['AMT', 'Riskfree'], weights):
        print(f"{ticker}: {w:.4f}")

    # Plot weights
    plot_weights(weights, ['AMT', 'RiskFree'])

if __name__ == "__main__":
    main()

   