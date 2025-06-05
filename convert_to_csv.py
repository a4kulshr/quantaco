import json
import pandas as pd
from datetime import datetime

# Read the JSON file
with open('stockprices.json', 'r') as f:
    data = json.load(f)

# Extract the time series data
time_series = data['Time Series (Daily)']

# Create lists to store the data
dates = []
prices = []

# Extract dates and close prices
for date, values in time_series.items():
    dates.append(date)
    prices.append(float(values['4. close']))

# Create DataFrame
df = pd.DataFrame({
    'Time': range(len(dates)),
    'Path': 1,  # Using 1 as we have only one path for historical data
    'Price': prices
})

# Save to CSV
df.to_csv('historical_prices.csv', index=False) 