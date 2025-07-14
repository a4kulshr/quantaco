# Advanced Arbitrage Detection for Political Trade Surges
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns

def detect_political_arbitrage_opportunities(stocks, politician_metadata, start_date='2020-01-01', end_date='2025-01-01'):
    """
    Detect arbitrage opportunities caused by political trade surges using politician metadata
    """
    print(f"Downloading data for {stocks} from {start_date} to {end_date}...")
    print(f"Using politician metadata: {politician_metadata}")
    
    # Download comprehensive data including volume
    df = yf.download(stocks, start=start_date, end=end_date)
    
    # Calculate various metrics
    price_data = df['Close']
    volume_data = df['Volume']
    returns = price_data.pct_change().dropna()
    
    # Encode politician metadata
    politician_encoder = LabelEncoder()
    party_encoder = LabelEncoder()
    sector_encoder = LabelEncoder()
    volume_encoder = LabelEncoder()
    
    # Extract unique values for encoding
    politicians = [metadata['politician'] for metadata in politician_metadata.values()]
    parties = [metadata['party'] for metadata in politician_metadata.values()]
    sectors = [metadata['sector'] for metadata in politician_metadata.values()]
    volumes = [metadata['volume'] for metadata in politician_metadata.values()]
    
    # Fit encoders
    politician_encoder.fit(politicians)
    party_encoder.fit(parties)
    sector_encoder.fit(sectors)
    volume_encoder.fit(volumes)
    
    # Calculate rolling statistics for anomaly detection
    window = 20  # 20-day rolling window
    
    # 1. Volatility spikes (sudden increases in price volatility)
    rolling_volatility = returns.rolling(window=window).std()
    volatility_threshold = rolling_volatility.rolling(window=window).mean() + 2 * rolling_volatility.rolling(window=window).std()
    volatility_spikes = rolling_volatility > volatility_threshold
    
    # 2. Volume surges (unusual trading volume)
    rolling_volume = volume_data.rolling(window=window).mean()
    volume_threshold = rolling_volume + 2 * volume_data.rolling(window=window).std()
    volume_surges = volume_data > volume_threshold
    
    # 3. Price momentum (sudden price movements)
    price_momentum = returns.rolling(window=5).sum()  # 5-day momentum
    momentum_threshold = price_momentum.rolling(window=window).mean() + 2 * price_momentum.rolling(window=window).std()
    momentum_spikes = price_momentum > momentum_threshold
    
    # 4. Cross-stock correlation breakdown (arbitrage opportunity indicator)
    rolling_corr = returns.rolling(window=window).corr()
    avg_correlation = rolling_corr.groupby(level=0).mean()
    correlation_breakdown = avg_correlation < avg_correlation.rolling(window=window).mean() - avg_correlation.rolling(window=window).std()
    
    # 5. Enhanced political trade surge detection using metadata
    political_surge_indicators = pd.DataFrame(index=returns.index)
    
    for stock in stocks:
        if stock in politician_metadata:
            metadata = politician_metadata[stock]
            
            # Encode politician features
            politician_encoded = politician_encoder.transform([metadata['politician']])[0]
            party_encoded = party_encoder.transform([metadata['party']])[0]
            sector_encoded = sector_encoder.transform([metadata['sector']])[0]
            volume_encoded = volume_encoder.transform([metadata['volume']])[0]
            
            # Create politician-specific surge indicators
            # Republican politicians might have different trading patterns
            party_factor = 1.2 if metadata['party'] == 'Republican' else 1.0
            
            # Technology sector might be more volatile
            sector_factor = 1.3 if metadata['sector'] == 'Technology' else 1.0
            
            # Volume category affects sensitivity
            volume_factor = 1.5 if metadata['volume'] == '1K-5K' else 1.0
            
            # Combine multiple indicators with politician metadata
            stock_indicators = (
                volatility_spikes[stock] * 0.25 * party_factor * sector_factor +
                volume_surges[stock] * 0.25 * volume_factor +
                momentum_spikes[stock] * 0.2 * sector_factor +
                (returns[stock] > returns[stock].rolling(window=window).mean() + 2 * returns[stock].rolling(window=window).std()) * 0.15 +
                # Add politician-specific patterns
                (volume_data[stock] > volume_data[stock].rolling(window=window).mean() * 1.5) * 0.15 * volume_factor
            )
            political_surge_indicators[stock] = stock_indicators
        else:
            # Fallback for stocks without politician metadata
            stock_indicators = (
                volatility_spikes[stock] * 0.3 +
                volume_surges[stock] * 0.3 +
                momentum_spikes[stock] * 0.2 +
                (returns[stock] > returns[stock].rolling(window=window).mean() + 2 * returns[stock].rolling(window=window).std()) * 0.2
            )
            political_surge_indicators[stock] = stock_indicators
    
    # 6. Enhanced arbitrage opportunity scoring with politician context
    arbitrage_scores = pd.DataFrame(index=returns.index)
    
    for stock in stocks:
        if stock in politician_metadata:
            metadata = politician_metadata[stock]
            
            # Politician-specific scoring
            politician_encoded = politician_encoder.transform([metadata['politician']])[0]
            party_encoded = party_encoder.transform([metadata['party']])[0]
            sector_encoded = sector_encoder.transform([metadata['sector']])[0]
            volume_encoded = volume_encoder.transform([metadata['volume']])[0]
            
            # Adjust thresholds based on politician characteristics
            party_threshold = 2.5 if metadata['party'] == 'Republican' else 2.0
            sector_threshold = 2.5 if metadata['sector'] == 'Technology' else 2.0
            volume_threshold = 2.5 if metadata['volume'] == '1K-5K' else 2.0
            
            # Calculate arbitrage score with politician context
            score = (
                political_surge_indicators[stock] * 0.4 +
                (returns[stock].abs() > returns[stock].rolling(window=window).std() * party_threshold) * 0.25 +
                (volume_data[stock] > volume_data[stock].rolling(window=window).mean() * volume_threshold) * 0.2 +
                # Add sector-specific volatility
                (rolling_volatility[stock] > rolling_volatility[stock].rolling(window=window).mean() * sector_threshold) * 0.15
            )
            arbitrage_scores[stock] = score
        else:
            # Standard scoring for stocks without metadata
            score = (
                political_surge_indicators[stock] * 0.4 +
                (returns[stock].abs() > returns[stock].rolling(window=window).std() * 3) * 0.3 +
                (volume_data[stock] > volume_data[stock].rolling(window=window).mean() * 2) * 0.3
            )
            arbitrage_scores[stock] = score
    
    # 7. Identify high-probability arbitrage periods
    high_arbitrage_periods = arbitrage_scores > 0.7  # Threshold for high probability
    
    return {
        'returns': returns,
        'volume_data': volume_data,
        'volatility_spikes': volatility_spikes,
        'volume_surges': volume_surges,
        'momentum_spikes': momentum_spikes,
        'political_surge_indicators': political_surge_indicators,
        'arbitrage_scores': arbitrage_scores,
        'high_arbitrage_periods': high_arbitrage_periods,
        'price_data': price_data,
        'politician_metadata': politician_metadata,
        'encoders': {
            'politician': politician_encoder,
            'party': party_encoder,
            'sector': sector_encoder,
            'volume': volume_encoder
        }
    }

def visualize_arbitrage_analysis(arbitrage_results, stocks):
    """Comprehensive visualization of arbitrage analysis results"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Price movements with arbitrage periods highlighted
    ax1 = plt.subplot(4, 2, 1)
    for stock in stocks:
        ax1.plot(arbitrage_results['price_data'].index, arbitrage_results['price_data'][stock], 
                label=stock, alpha=0.7, linewidth=1.5)
        
        # Highlight arbitrage periods
        stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
        if stock_arbitrage_periods.any():
            arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
            if len(arbitrage_dates) > 0:
                arbitrage_prices = arbitrage_results['price_data'].loc[arbitrage_dates, stock]
                ax1.scatter(arbitrage_dates, arbitrage_prices, color='red', s=80, alpha=0.8, 
                           marker='^', edgecolors='black', linewidth=1)

    ax1.set_title('Stock Prices with Political Arbitrage Periods Highlighted', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Arbitrage scores over time
    ax2 = plt.subplot(4, 2, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, stock in enumerate(stocks):
        ax2.plot(arbitrage_results['arbitrage_scores'].index, arbitrage_results['arbitrage_scores'][stock], 
                label=stock, alpha=0.8, color=colors[i], linewidth=1.5)
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Arbitrage Threshold', linewidth=2)
    ax2.set_title('Political Arbitrage Scores Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Arbitrage Score', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Volume surges
    ax3 = plt.subplot(4, 2, 3)
    for i, stock in enumerate(stocks):
        ax3.plot(arbitrage_results['volume_data'].index, arbitrage_results['volume_data'][stock], 
                label=stock, alpha=0.7, color=colors[i], linewidth=1)
    ax3.set_title('Trading Volume', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Volume', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Volatility spikes
    ax4 = plt.subplot(4, 2, 4)
    for i, stock in enumerate(stocks):
        rolling_vol = arbitrage_results['returns'][stock].rolling(window=20).std()
        ax4.plot(rolling_vol.index, rolling_vol, label=stock, alpha=0.7, color=colors[i], linewidth=1.5)
    ax4.set_title('Rolling Volatility (20-day window)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Volatility', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Political surge indicators
    ax5 = plt.subplot(4, 2, 5)
    for i, stock in enumerate(stocks):
        ax5.plot(arbitrage_results['political_surge_indicators'].index, 
                arbitrage_results['political_surge_indicators'][stock], 
                label=stock, alpha=0.8, color=colors[i], linewidth=1.5)
    ax5.set_title('Political Trade Surge Indicators', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=12)
    ax5.set_ylabel('Surge Indicator', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Returns distribution with arbitrage periods
    ax6 = plt.subplot(4, 2, 6)
    arbitrage_returns = []
    normal_returns = []

    for stock in stocks:
        stock_returns = arbitrage_results['returns'][stock]
        stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
        
        if stock_arbitrage_periods.any():
            arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
            
            for date in stock_returns.index:
                if date in arbitrage_dates:
                    arbitrage_returns.append(stock_returns[date])
                else:
                    normal_returns.append(stock_returns[date])
        else:
            # If no arbitrage periods, all returns are normal
            normal_returns.extend(stock_returns.values)

    ax6.hist(normal_returns, bins=50, alpha=0.7, label='Normal Returns', density=True, color='blue')
    ax6.hist(arbitrage_returns, bins=20, alpha=0.7, label='Arbitrage Period Returns', density=True, color='red')
    ax6.set_title('Returns Distribution: Normal vs Arbitrage Periods', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Returns', fontsize=12)
    ax6.set_ylabel('Density', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # 7. Politician-specific analysis
    ax7 = plt.subplot(4, 2, 7)
    politician_stats = {}
    
    for stock in stocks:
        if stock in arbitrage_results['politician_metadata']:
            metadata = arbitrage_results['politician_metadata'][stock]
            politician_key = f"{metadata['politician']} ({metadata['party']})"
            
            if politician_key not in politician_stats:
                politician_stats[politician_key] = {'scores': [], 'returns': []}
            
            stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
            if stock_arbitrage_periods.any():
                arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
                if len(arbitrage_dates) > 0:
                    scores = arbitrage_results['arbitrage_scores'].loc[arbitrage_dates, stock]
                    returns = arbitrage_results['returns'].loc[arbitrage_dates, stock]
                    politician_stats[politician_key]['scores'].extend(scores.values)
                    politician_stats[politician_key]['returns'].extend(returns.values)
    
    # Plot politician performance
    politicians = list(politician_stats.keys())
    avg_scores = [np.mean(stats['scores']) if stats['scores'] else 0 for stats in politician_stats.values()]
    avg_returns = [np.mean(stats['returns']) if stats['returns'] else 0 for stats in politician_stats.values()]
    
    x = np.arange(len(politicians))
    width = 0.35
    
    ax7.bar(x - width/2, avg_scores, width, label='Avg Arbitrage Score', alpha=0.8)
    ax7.bar(x + width/2, avg_returns, width, label='Avg Return During Arbitrage', alpha=0.8)
    
    ax7.set_title('Politician Performance Analysis', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Politician (Party)', fontsize=12)
    ax7.set_ylabel('Score/Return', fontsize=12)
    ax7.set_xticks(x)
    ax7.set_xticklabels(politicians, rotation=45, ha='right')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # 8. Sector analysis
    ax8 = plt.subplot(4, 2, 8)
    sector_stats = {}
    
    for stock in stocks:
        if stock in arbitrage_results['politician_metadata']:
            sector = arbitrage_results['politician_metadata'][stock]['sector']
            
            if sector not in sector_stats:
                sector_stats[sector] = {'scores': [], 'returns': []}
            
            stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
            if stock_arbitrage_periods.any():
                arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
                if len(arbitrage_dates) > 0:
                    scores = arbitrage_results['arbitrage_scores'].loc[arbitrage_dates, stock]
                    returns = arbitrage_results['returns'].loc[arbitrage_dates, stock]
                    sector_stats[sector]['scores'].extend(scores.values)
                    sector_stats[sector]['returns'].extend(returns.values)
    
    sectors = list(sector_stats.keys())
    sector_avg_scores = [np.mean(stats['scores']) if stats['scores'] else 0 for stats in sector_stats.values()]
    sector_avg_returns = [np.mean(stats['returns']) if stats['returns'] else 0 for stats in sector_stats.values()]
    
    x = np.arange(len(sectors))
    
    ax8.bar(x - width/2, sector_avg_scores, width, label='Avg Arbitrage Score', alpha=0.8)
    ax8.bar(x + width/2, sector_avg_returns, width, label='Avg Return During Arbitrage', alpha=0.8)
    
    ax8.set_title('Sector Performance Analysis', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Sector', fontsize=12)
    ax8.set_ylabel('Score/Return', fontsize=12)
    ax8.set_xticks(x)
    ax8.set_xticklabels(sectors, rotation=45, ha='right')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_heatmap_visualization(arbitrage_results, stocks):
    """Create heatmap visualization of arbitrage patterns"""
    
    # Create correlation heatmap of arbitrage scores
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix of arbitrage scores
    score_corr = arbitrage_results['arbitrage_scores'].corr()
    
    # Create heatmap
    sns.heatmap(score_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Political Arbitrage Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Stocks', fontsize=12)
    plt.ylabel('Stocks', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_timeline_visualization(arbitrage_results, stocks):
    """Create timeline visualization of arbitrage events"""
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define colors for different stocks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, stock in enumerate(stocks):
        stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
        if stock_arbitrage_periods.any():
            arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
            if len(arbitrage_dates) > 0:
                scores = arbitrage_results['arbitrage_scores'].loc[arbitrage_dates, stock]
                
                # Plot timeline
                ax.scatter(arbitrage_dates, [i] * len(arbitrage_dates), 
                          c=scores, cmap='viridis', s=100, alpha=0.8, 
                          edgecolors='black', linewidth=1)
    
    # Customize plot
    ax.set_yticks(range(len(stocks)))
    ax.set_yticklabels(stocks)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    ax.set_title('Timeline of Political Arbitrage Events', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Arbitrage Score', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def print_arbitrage_analysis(arbitrage_results, stocks):
    """Print detailed arbitrage analysis with politician context"""
    print("=== POLITICAL ARBITRAGE OPPORTUNITY ANALYSIS ===")

    # Summary statistics
    for stock in stocks:
        stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
        if stock_arbitrage_periods.any():
            arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
            if len(arbitrage_dates) > 0:
                # Get politician metadata
                politician_info = ""
                if stock in arbitrage_results['politician_metadata']:
                    metadata = arbitrage_results['politician_metadata'][stock]
                    politician_info = f" ({metadata['politician']}, {metadata['party']}, {metadata['sector']}, {metadata['volume']})"
                
                print(f"\n{stock}{politician_info} - Political Arbitrage Opportunities:")
                print(f"  Number of arbitrage periods: {len(arbitrage_dates)}")
                print(f"  Average arbitrage score: {arbitrage_results['arbitrage_scores'][stock].mean():.4f}")
                print(f"  Max arbitrage score: {arbitrage_results['arbitrage_scores'][stock].max():.4f}")
                
                # Calculate returns during arbitrage periods
                arbitrage_returns_stock = arbitrage_results['returns'].loc[arbitrage_dates, stock]
                print(f"  Average return during arbitrage: {arbitrage_returns_stock.mean():.4f}")
                print(f"  Volatility during arbitrage: {arbitrage_returns_stock.std():.4f}")
                
                # Show top arbitrage periods
                top_periods = arbitrage_results['arbitrage_scores'][stock].nlargest(3)
                print(f"  Top 3 arbitrage periods:")
                for date, score in top_periods.items():
                    return_val = arbitrage_results['returns'].loc[date, stock]
                    volume_val = arbitrage_results['volume_data'].loc[date, stock]
                    print(f"    {date.strftime('%Y-%m-%d')}: Score={score:.4f}, Return={return_val:.4f}, Volume={volume_val:,.0f}")

def main():
    """Main function to run the political arbitrage detection"""
    # Define stocks with political context
    stocks = ['META', 'NFLX', 'NOW', 'FLL']
    
    # Define politician metadata from your notebook
    politician_metadata = {
        'META': {'politician': 'John McGuire', 'party': 'Republican', 'sector': 'Technology', 'volume': '1K-5K'},
        'NFLX': {'politician': 'John McGuire', 'party': 'Republican', 'sector': 'Technology', 'volume': '1K-5K'},
        'NOW': {'politician': 'John McGuire', 'party': 'Republican', 'sector': 'Technology', 'volume': '1K-5K'},
        'FLL': {'politician': 'Susie Lee', 'party': 'Democratic', 'sector': 'Gaming', 'volume': 'Low'}
    }

    print("Starting Political Arbitrage Detection...")
    
    # Run the detection
    arbitrage_results = detect_political_arbitrage_opportunities(stocks, politician_metadata)
    
    # Print analysis
    print_arbitrage_analysis(arbitrage_results, stocks)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_arbitrage_analysis(arbitrage_results, stocks)
    create_heatmap_visualization(arbitrage_results, stocks)
    create_timeline_visualization(arbitrage_results, stocks)
    
    print("\nPolitical arbitrage detection complete!")

if __name__ == "__main__":
    main() 