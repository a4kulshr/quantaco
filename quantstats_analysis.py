import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_stock_data():
    """Load stock data from existing files"""
    try:
        # Load from historical_prices.csv
        df = pd.read_csv('historical_prices.csv')
        df['Date'] = pd.to_datetime(df['Time'], unit='D', origin='2023-01-01')
        df.set_index('Date', inplace=True)
        
        # Calculate returns
        returns = df['Price'].pct_change().dropna()
        
        return returns
    except Exception as e:
        logging.error(f"Error loading historical_prices.csv: {str(e)}")
        
        # Fallback to stockprices.json
        try:
            with open('stockprices.json', 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            prices = []
            dates = []
            for date, values in data['Time Series (Daily)'].items():
                prices.append(float(values['4. close']))
                dates.append(pd.to_datetime(date))
            
            df = pd.DataFrame({'Price': prices}, index=dates)
            df.sort_index(inplace=True)
            
            # Calculate returns
            returns = df['Price'].pct_change().dropna()
            
            return returns
        except Exception as e:
            logging.error(f"Error loading stockprices.json: {str(e)}")
            raise

def calculate_metrics(returns):
    """Calculate key performance metrics"""
    # Calculate total return and CAGR
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Calculate Sharpe Ratio
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    
    # Calculate drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate additional metrics
    win_rate = len(returns[returns > 0]) / len(returns)
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility,
        'Win Rate': win_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Profit Factor': profit_factor
    }

def generate_plots(returns, save_path='amd_analysis_plots.png'):
    """Generate and save analysis plots"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    # Plot cumulative returns
    cum_returns = (1 + returns).cumprod()
    axes[0].plot(cum_returns.index, cum_returns.values)
    axes[0].set_title('Cumulative Returns')
    axes[0].grid(True)
    
    # Plot rolling volatility (20-day window)
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
    axes[1].plot(rolling_vol.index, rolling_vol.values)
    axes[1].set_title('20-day Rolling Volatility (Annualized)')
    axes[1].grid(True)
    
    # Plot drawdown
    drawdown = (cum_returns / cum_returns.cummax()) - 1
    axes[2].plot(drawdown.index, drawdown.values)
    axes[2].set_title('Drawdown')
    axes[2].grid(True)
    
    # Plot monthly returns heatmap
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns = monthly_returns.to_frame()
    monthly_returns.index = pd.MultiIndex.from_arrays([
        monthly_returns.index.year,
        monthly_returns.index.month
    ])
    monthly_returns = monthly_returns.unstack()
    
    sns.heatmap(monthly_returns, 
                annot=True, 
                fmt='.2%', 
                cmap='RdYlGn', 
                center=0,
                ax=axes[3])
    axes[3].set_title('Monthly Returns Heatmap')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_report(metrics, returns, output_file='amd_analysis.html'):
    """Generate HTML report with metrics and analysis"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AMD Stock Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .metric-card {{ 
                background: #f5f5f5; 
                padding: 15px; 
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; }}
            .plot {{ margin: 20px 0; }}
            h1 {{ color: #2c3e50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AMD Stock Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Metrics</h2>
            <div class="metrics">
    """
    
    # Add metrics to HTML
    for label, value in metrics.items():
        if isinstance(value, float):
            if 'Rate' in label or 'Factor' in label:
                formatted_value = f"{value:.2%}"
            else:
                formatted_value = f"{value:.2%}" if 'Return' in label or 'Drawdown' in label else f"{value:.2f}"
        else:
            formatted_value = str(value)
            
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>Analysis Plots</h2>
            <div class="plot">
                <img src="amd_analysis_plots.png" alt="Analysis Plots" style="width: 100%;">
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def analyze_stock():
    try:
        # Get stock data from existing files
        logging.info("Loading stock data from existing files...")
        returns = load_stock_data()
        
        # Calculate metrics
        logging.info("Calculating performance metrics...")
        metrics = calculate_metrics(returns)
        
        # Generate plots
        logging.info("Generating analysis plots...")
        generate_plots(returns)
        
        # Generate HTML report
        logging.info("Generating HTML report...")
        generate_html_report(metrics, returns)
        
        # Save metrics to CSV
        pd.Series(metrics).to_csv('amd_metrics.csv')
        
        logging.info("Analysis complete! Check amd_analysis.html for the full report.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_stock() 