import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt

amt = yf.Ticker("AMT")  # Note the capital 'T' in Ticker

# Fetch 1 year of historical daily price data
amt_data = amt.history(period="1y")

print(amt_data.head())

