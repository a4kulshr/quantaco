import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Configure matplotlib settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# Read the CSV data
df = pd.read_csv('historical_prices.csv')

# Set the style
plt.style.use('seaborn-v0_8-darkgrid')

# Create the plot with a modern color scheme
fig, ax = plt.subplots(figsize=(15, 8))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')

# Plot the historical price path with enhanced styling
ax.plot(df['Time'], df['Price'], color='#2E86C1', linewidth=2.5, label='Historical Prices', alpha=0.9)

# Add moving averages with improved styling
df['MA20'] = df['Price'].rolling(window=20).mean()
df['MA50'] = df['Price'].rolling(window=50).mean()
ax.plot(df['Time'], df['MA20'], color='#E74C3C', linestyle='--', linewidth=1.5, label='20-day MA', alpha=0.8)
ax.plot(df['Time'], df['MA50'], color='#27AE60', linestyle='--', linewidth=1.5, label='50-day MA', alpha=0.8)

# Customize the plot
ax.set_title('AMT Stock Historical Prices with Moving Averages', pad=20)
ax.set_xlabel('Trading Days', labelpad=10)
ax.set_ylabel('Stock Price ($)', labelpad=10)
ax.grid(True, alpha=0.2)
ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)

# Add a subtle border
for spine in ax.spines.values():
    spine.set_color('#dcdcdc')

# Save the plot with high quality
plt.savefig('historical_prices.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# Create a histogram of price changes
plt.figure(figsize=(12, 7))
fig.patch.set_facecolor('#f8f9fa')
ax = plt.gca()
ax.set_facecolor('#ffffff')

price_changes = df['Price'].pct_change().dropna()
sns.histplot(price_changes, bins=50, kde=True, color='#2E86C1', alpha=0.7)
plt.title('Distribution of Daily Price Changes', pad=20)
plt.xlabel('Daily Return', labelpad=10)
plt.ylabel('Frequency', labelpad=10)

# Add mean and std annotations
mean_return = price_changes.mean()
std_return = price_changes.std()
plt.axvline(mean_return, color='#E74C3C', linestyle='--', alpha=0.8)
plt.text(mean_return, plt.ylim()[1]*0.9, 
         f'μ = {mean_return:.2%}',
         color='#E74C3C', ha='center')

# Add grid and styling
plt.grid(True, alpha=0.2)
for spine in plt.gca().spines.values():
    spine.set_color('#dcdcdc')

plt.savefig('price_changes_distribution.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# Create a volatility plot
plt.figure(figsize=(15, 7))
fig.patch.set_facecolor('#f8f9fa')
ax = plt.gca()
ax.set_facecolor('#ffffff')

volatility = price_changes.rolling(window=20).std() * (252 ** 0.5)  # Annualized volatility
ax.plot(df['Time'][1:], volatility, color='#27AE60', linewidth=2.5, alpha=0.9)
ax.set_title('20-day Rolling Volatility (Annualized)', pad=20)
ax.set_xlabel('Trading Days', labelpad=10)
ax.set_ylabel('Volatility (σ)', labelpad=10)
ax.grid(True, alpha=0.2)

# Add mean volatility line
mean_vol = volatility.mean()
ax.axhline(mean_vol, color='#E74C3C', linestyle='--', alpha=0.8)
ax.text(len(volatility)*0.02, mean_vol, 
        f'σ̄ = {mean_vol:.2%}',
        color='#E74C3C', va='bottom')

# Add styling
for spine in ax.spines.values():
    spine.set_color('#dcdcdc')

plt.savefig('volatility.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close() 