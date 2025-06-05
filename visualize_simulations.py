import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read the CSV data
df = pd.read_csv('historical_prices.csv')

# Set the style
plt.style.use('seaborn-v0_8')

# Create the plot
plt.figure(figsize=(15, 8))

# Plot the historical price path
plt.plot(df['Time'], df['Price'], 'b-', linewidth=2, label='Historical Prices')

# Add moving averages
df['MA20'] = df['Price'].rolling(window=20).mean()
df['MA50'] = df['Price'].rolling(window=50).mean()
plt.plot(df['Time'], df['MA20'], 'r--', linewidth=1.5, label='20-day MA')
plt.plot(df['Time'], df['MA50'], 'g--', linewidth=1.5, label='50-day MA')

# Customize the plot
plt.title('AMT Stock Historical Prices with Moving Averages')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price ($)')
plt.grid(True, alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('historical_prices.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a histogram of price changes
plt.figure(figsize=(10, 6))
price_changes = df['Price'].pct_change().dropna()
sns.histplot(price_changes, bins=50, kde=True)
plt.title('Distribution of Daily Price Changes')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.savefig('price_changes_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a volatility plot
plt.figure(figsize=(12, 6))
volatility = price_changes.rolling(window=20).std() * (252 ** 0.5)  # Annualized volatility
plt.plot(df['Time'][1:], volatility, 'g-', linewidth=2)
plt.title('20-day Rolling Volatility (Annualized)')
plt.xlabel('Trading Days')
plt.ylabel('Volatility')
plt.grid(True, alpha=0.3)
plt.savefig('volatility.png', dpi=300, bbox_inches='tight')
plt.close() 