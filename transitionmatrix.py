import yfinance as yf
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
# set return index
ticker = ['AMT']

from curl_cffi import requests

session = requests.Session(impersonate="chrome")
ticker = yf.Ticker('AMT', session=session)
risk_free = yf.Ticker('^GSPC', session=session)

df = yf.download("AMT", start="2023-01-01", end="2024-01-01")
amt_data = ticker.history(period="1y")  # Last 1 year
sp500_data = risk_free.history(start="2023-01-01", end="2024-01-01")  # Custom range
return_df= amt_data[['Close']].reset_index()
return_df.columns = ['Date', 'Close']
riskfree_df = sp500_data[['Close']].reset_index()
riskfree_df.columns = ['Date', 'Close']

# calculate daily returns
return_df['Return'] = return_df['Close'].pct_change()
riskfree_df['Return'] = riskfree_df['Close'].pct_change()

returns_clean = return_df.dropna(subset = ['Return']).copy()
returns_reshaped = returns_clean['Return'].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=42)
returns_clean['Cluster'] = kmeans.fit_predict(returns_reshaped)

#set bull bear and base indexes
centers = kmeans.cluster_centers_
sorted_centers = np.sort(centers, axis=0)
states = {
    sorted_centers[0][0]: 'Bear',
    sorted_centers[1][0]: 'Base',
    sorted_centers[2][0]: 'Bull'

}
returns_clean['State'] = returns_clean['Cluster'].map(lambda x: states[centers[x][0]])

#compute transition matrix
transition_matrix = pd.crosstab(returns_clean['State'].shift(1), returns_clean['State'], normalize='index')

# Plot the transition matrix
plt.figure(figsize=(8, 6))
sn.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Transition Probability'})
plt.title('Transition Matrix')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()