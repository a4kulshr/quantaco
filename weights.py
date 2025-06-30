import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_weights_and_sharpe(returns, sigma=0.04):
    mu = returns.mean()
    S = returns.cov()
    S = (S + S.T) / 2
    inv_S = inv(S)
    w = inv_S.dot(mu)
    theta = np.dot(mu, w)
    w_optimal = sigma/np.sqrt(theta) * w
    # No shorting, normalize to sum to 1
    w_optimal = np.maximum(w_optimal, 0)
    if w_optimal.sum() > 0:
        w_optimal = w_optimal / w_optimal.sum()
    else:
        w_optimal = np.ones_like(w_optimal) / len(w_optimal)
    sharpe_ratio = np.sqrt(252) * np.dot(w_optimal, mu) / np.sqrt(np.dot(w_optimal, S.dot(w_optimal)))
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
    df = df[['Date', 'Close_x', 'Close_y']].dropna()
    df = df.rename(columns={'Close_x': 'AMT', 'Close_y': 'CCI'})
    df.set_index('Date', inplace=True)

    # Calculate daily returns
    returns = df.pct_change().dropna()

    # Calculate weights and Sharpe ratio
    weights, sharpe = calculate_weights_and_sharpe(returns, sigma=0.04)

    # Print results
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
    print("Portfolio Weights:")
    for ticker, w in zip(['AMT', 'CCI'], weights):
        print(f"{ticker}: {w:.4f}")

    # Plot weights
    plot_weights(weights, ['AMT', 'CCI'])

if __name__ == "__main__":
    main()