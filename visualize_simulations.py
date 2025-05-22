import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Read the CSV data from stdin
df = pd.read_csv(sys.stdin)

# Set the style
plt.style.use('seaborn-v0_8')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot each path
for path in df['Path'].unique():
    path_data = df[df['Path'] == path]
    plt.plot(path_data['Time'], path_data['Price'], alpha=0.3, linewidth=1)

# Add mean path
mean_prices = df.groupby('Time')['Price'].mean()
plt.plot(mean_prices.index, mean_prices.values, 'k--', linewidth=2, label='Mean Path')

# Customize the plot
plt.title('Geometric Brownian Motion Simulation Paths for AMT Stock')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price ($)')
plt.grid(True, alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('gbm_simulation.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a histogram of final prices
plt.figure(figsize=(10, 6))
final_prices = df[df['Time'] == df['Time'].max()]['Price']
sns.histplot(final_prices, bins=30, kde=True)
plt.title('Distribution of Final Prices')
plt.xlabel('Final Price ($)')
plt.ylabel('Frequency')
plt.savefig('final_prices_distribution.png', dpi=300, bbox_inches='tight')
plt.close() 