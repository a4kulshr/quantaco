

import sys
print(sys.executable)
print(sys.version)
import subprocess
#subprocess.run(["pip", "show", "quantstats"])
#subprocess.run([sys.executable, "-m", "pip", "install", "quantstats"])

import quantstats as qs

# Extend pandas functionality
qs.extend_pandas()

# Fetch the daily returns for a stock
stock = qs.utils.download_returns('AMT').dropna()  # Ensure data is clean

# Show Sharpe ratio
print("Sharpe Ratio:", stock.sharpe())

# Generate snapshot plot
qs.plots.snapshot(stock, title='AMT Performance', show=True)

# Generate HTML report
qs.reports.html(stock, output='AMT_report.html')