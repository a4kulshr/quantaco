import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Read data from CSV
stock_data = pd.read_csv('AMT CCI Data.csv', parse_dates=['Date'])
stock_data.set_index('Date', inplace=True)

# Use Close_x for AMT and Close_y for CCI
stock_data = stock_data[['Close_x', 'Close_y']].dropna()
stock_data.columns = ['AMT', 'CCI']

# OLS hedge ratio
X = sm.add_constant(stock_data['AMT'])
Y = stock_data['CCI']
model = sm.OLS(Y, X).fit()
hedge_ratio = model.params['AMT']

# Cointegration test
score, p_value, _ = coint(stock_data['AMT'], stock_data['CCI'])

# Calculate spread
spread = stock_data['AMT'] - hedge_ratio * stock_data['CCI']
stock_data['Spread'] = spread

# --- Professional Plotting ---
sns.set(style='whitegrid', context='talk', font_scale=1.1)
plt.figure(figsize=(16, 8))

# Color by deviation from mean
mean_spread = spread.mean()
std_spread = spread.std()
colors = np.where(spread > mean_spread, '#2a9d8f', '#e76f51')

plt.plot(stock_data.index, spread, color='#264653', lw=2, label='Spread (AMT - β·CCI)')
plt.axhline(mean_spread, color='#e63946', linestyle='--', lw=2, label='Mean Spread')
plt.fill_between(stock_data.index, mean_spread-std_spread, mean_spread+std_spread, color='#a8dadc', alpha=0.3, label='±1 Std Dev')

# Title and subtitle
plt.title('Spread between AMT and CCI', fontsize=24, fontweight='bold', pad=20)
plt.suptitle(r"Spread = AMT $-$ $\beta$·CCI   |   OLS $\beta$ = {:.4f}   |   Cointegration p-value = {:.4g}".format(hedge_ratio, p_value),
             fontsize=16, y=0.92, color='#555')

plt.xlabel('Date', fontsize=16)
plt.ylabel('Spread', fontsize=16)
plt.legend(fontsize=14, frameon=True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('spread_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Professional spread plot 'spread_plot.png' has been generated.") 