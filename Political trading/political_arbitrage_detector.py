# Advanced Arbitrage Detection for Political Trade Surges
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.neural_network import MLPClassifier

def train_arbitrage_neural_network(arbitrage_results, stocks):
    """
    Train a neural network to recognize arbitrage patterns
    """
    print("Training neural network for arbitrage pattern recognition...")
    
    # Prepare features for neural network
    features_list = []
    labels_list = []
    
    for stock in stocks:
        # Get stock data
        returns = arbitrage_results['returns'][stock]
        volume_data = arbitrage_results['volume_data'][stock]
        political_surge = arbitrage_results['political_surge_indicators'][stock]
        arbitrage_scores = arbitrage_results['arbitrage_scores'][stock]
        
        # Create features (last 5 periods for pattern recognition)
        for i in range(5, len(returns)):
            # Price features
            price_features = [
                returns.iloc[i-1], returns.iloc[i-2], returns.iloc[i-3], returns.iloc[i-4], returns.iloc[i-5],
                volume_data.iloc[i-1] / volume_data.iloc[i-2] if volume_data.iloc[i-2] > 0 else 1,  # Volume ratio
                political_surge.iloc[i-1], political_surge.iloc[i-2]
            ]
            
            # Add politician metadata features
            if stock in arbitrage_results['politician_metadata']:
                metadata = arbitrage_results['politician_metadata'][stock]
                politician_features = [
                    1 if metadata['party'] == 'Republican' else 0,
                    1 if metadata['sector'] == 'Technology' else 0,
                    1 if metadata['volume'] == '1K-5K' else 0
                ]
            else:
                politician_features = [0, 0, 0]
            
            # Combine all features
            all_features = price_features + politician_features
            features_list.append(all_features)
            
            # Create label: 1 for arbitrage period, 0 for normal
            is_arbitrage = 1 if arbitrage_scores.iloc[i] > 0.7 else 0
            labels_list.append(is_arbitrage)
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train neural network
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train, y_train)
    
    # Evaluate
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    
    print(f"Neural Network Training Results:")
    print(f"  Training Accuracy: {train_score:.4f}")
    print(f"  Test Accuracy: {test_score:.4f}")
    
    return mlp, X, y

def predict_arbitrage_with_nn(arbitrage_results, stocks, mlp_model):
    """
    Use trained neural network to predict arbitrage periods
    """
    print("Predicting arbitrage periods with neural network...")
    
    nn_predictions = pd.DataFrame(index=arbitrage_results['returns'].index, columns=stocks)
    nn_probabilities = pd.DataFrame(index=arbitrage_results['returns'].index, columns=stocks)
    
    for stock in stocks:
        # Get stock data
        returns = arbitrage_results['returns'][stock]
        volume_data = arbitrage_results['volume_data'][stock]
        political_surge = arbitrage_results['political_surge_indicators'][stock]
        
        stock_predictions = []
        stock_probabilities = []
        
        for i in range(len(returns)):
            if i < 5:  
                stock_predictions.append(0)
                stock_probabilities.append(0.0)
            else:
                # Create features (same as training)
                price_features = [
                    returns.iloc[i-1], returns.iloc[i-2], returns.iloc[i-3], returns.iloc[i-4], returns.iloc[i-5],
                    volume_data.iloc[i-1] / volume_data.iloc[i-2] if volume_data.iloc[i-2] > 0 else 1,
                    political_surge.iloc[i-1], political_surge.iloc[i-2]
                ]
                
                # Add politician metadata features
                if stock in arbitrage_results['politician_metadata']:
                    metadata = arbitrage_results['politician_metadata'][stock]
                    politician_features = [
                        1 if metadata['party'] == 'Republican' else 0,
                        1 if metadata['sector'] == 'Technology' else 0,
                        1 if metadata['volume'] == '1K-5K' else 0
                    ]
                else:
                    politician_features = [0, 0, 0]
                
                all_features = price_features + politician_features
                
                # Predict
                prediction = mlp_model.predict([all_features])[0]   #mlp sets prediction to 1 or 0
                probability = mlp_model.predict_proba([all_features])[0][1]  # Probability of arbitrage
                
                stock_predictions.append(prediction)
                stock_probabilities.append(probability)
        
        nn_predictions[stock] = stock_predictions
        nn_probabilities[stock] = stock_probabilities
    
    return nn_predictions, nn_probabilities

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
    
    # 8. Train and use neural network for pattern recognition
    arbitrage_results = {
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
    
    # Train neural network
    mlp_model, X, y = train_arbitrage_neural_network(arbitrage_results, stocks)
    
    # Get neural network predictions
    nn_predictions, nn_probabilities = predict_arbitrage_with_nn(arbitrage_results, stocks, mlp_model)
    
    # Add neural network results to arbitrage_results
    arbitrage_results['nn_predictions'] = nn_predictions
    arbitrage_results['nn_probabilities'] = nn_probabilities
    arbitrage_results['mlp_model'] = mlp_model
    
    # Create combined arbitrage detection (rule-based OR neural network)
    combined_arbitrage = pd.DataFrame(index=returns.index, columns=stocks)
    for stock in stocks:
        combined_arbitrage[stock] = (
            (arbitrage_scores[stock] > 0.7) | (nn_probabilities[stock] > 0.7)
        )
    
    arbitrage_results['combined_arbitrage_periods'] = combined_arbitrage
    
    return arbitrage_results

def visualize_arbitrage_analysis(arbitrage_results, stocks):
    """Comprehensive visualization of arbitrage analysis results"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 28))
    
    # 1. Price movements with arbitrage periods highlighted
    ax1 = plt.subplot(5, 2, 1)
    for stock in stocks:
        ax1.plot(arbitrage_results['price_data'].index, arbitrage_results['price_data'][stock], 
                label=stock, alpha=0.7, linewidth=1.5)
        
        # Highlight rule-based arbitrage periods
        stock_arbitrage_periods = arbitrage_results['high_arbitrage_periods'][stock]
        if stock_arbitrage_periods.any():
            arbitrage_dates = stock_arbitrage_periods[stock_arbitrage_periods].index
            if len(arbitrage_dates) > 0:
                arbitrage_prices = arbitrage_results['price_data'].loc[arbitrage_dates, stock]
                ax1.scatter(arbitrage_dates, arbitrage_prices, color='red', s=80, alpha=0.8, 
                           marker='^', edgecolors='black', linewidth=1, label=f'{stock} Rule-Based' if stock == stocks[0] else "")
        
        # Highlight neural network arbitrage periods
        if 'nn_predictions' in arbitrage_results:
            nn_arbitrage_periods = arbitrage_results['nn_predictions'][stock] == 1
            if nn_arbitrage_periods.any():
                nn_dates = nn_arbitrage_periods[nn_arbitrage_periods].index
                if len(nn_dates) > 0:
                    nn_prices = arbitrage_results['price_data'].loc[nn_dates, stock]
                    ax1.scatter(nn_dates, nn_prices, color='green', s=60, alpha=0.8, 
                               marker='x', edgecolors='black', linewidth=1, label=f'{stock} NN' if stock == stocks[0] else "")

    ax1.set_title('Stock Prices with Arbitrage Periods (Red=Rule-Based, Green=Neural Network)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Arbitrage scores over time
    ax2 = plt.subplot(5, 2, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, stock in enumerate(stocks):
        ax2.plot(arbitrage_results['arbitrage_scores'].index, arbitrage_results['arbitrage_scores'][stock], 
                label=f'{stock} Rule-Based', alpha=0.8, color=colors[i], linewidth=1.5)
        
        # Add neural network probabilities
        if 'nn_probabilities' in arbitrage_results:
            ax2.plot(arbitrage_results['nn_probabilities'].index, arbitrage_results['nn_probabilities'][stock], 
                    label=f'{stock} NN', alpha=0.6, color=colors[i], linewidth=1, linestyle='--')
    
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Arbitrage Threshold', linewidth=2)
    ax2.set_title('Arbitrage Scores: Rule-Based vs Neural Network', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Score/Probability', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Volume surges
    ax3 = plt.subplot(5, 2, 3)
    for i, stock in enumerate(stocks):
        ax3.plot(arbitrage_results['volume_data'].index, arbitrage_results['volume_data'][stock], 
                label=stock, alpha=0.7, color=colors[i], linewidth=1)
    ax3.set_title('Trading Volume', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Volume', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Volatility spikes
    ax4 = plt.subplot(5, 2, 4)
    for i, stock in enumerate(stocks):
        rolling_vol = arbitrage_results['returns'][stock].rolling(window=20).std()
        ax4.plot(rolling_vol.index, rolling_vol, label=stock, alpha=0.7, color=colors[i], linewidth=1.5)
    ax4.set_title('Rolling Volatility (20-day window)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Volatility', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Political surge indicators
    ax5 = plt.subplot(5, 2, 5)
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
    ax6 = plt.subplot(5, 2, 6)
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

    # 7. Neural Network vs Rule-Based Comparison
    ax7 = plt.subplot(5, 2, 7)
    if 'nn_probabilities' in arbitrage_results:
        comparison_data = []
        labels = []
        
        for stock in stocks:
            # Rule-based arbitrage periods
            rule_based_count = arbitrage_results['high_arbitrage_periods'][stock].sum()
            # Neural network arbitrage periods
            nn_count = arbitrage_results['nn_predictions'][stock].sum()
            # Combined arbitrage periods
            combined_count = arbitrage_results['combined_arbitrage_periods'][stock].sum()
            
            comparison_data.append([rule_based_count, nn_count, combined_count])
            labels.append(stock)
        
        comparison_data = np.array(comparison_data)
        x = np.arange(len(labels))
        width = 0.25
        
        ax7.bar(x - width, comparison_data[:, 0], width, label='Rule-Based', alpha=0.8)
        ax7.bar(x, comparison_data[:, 1], width, label='Neural Network', alpha=0.8)
        ax7.bar(x + width, comparison_data[:, 2], width, label='Combined', alpha=0.8)
        
        ax7.set_title('Arbitrage Period Detection Comparison', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Stocks', fontsize=12)
        ax7.set_ylabel('Number of Arbitrage Periods', fontsize=12)
        ax7.set_xticks(x)
        ax7.set_xticklabels(labels)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)

    # 8. Politician-specific analysis
    ax8 = plt.subplot(5, 2, 8)
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
    
    ax8.bar(x - width/2, avg_scores, width, label='Avg Arbitrage Score', alpha=0.8)
    ax8.bar(x + width/2, avg_returns, width, label='Avg Return During Arbitrage', alpha=0.8)
    
    ax8.set_title('Politician Performance Analysis', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Politician (Party)', fontsize=12)
    ax8.set_ylabel('Score/Return', fontsize=12)
    ax8.set_xticks(x)
    ax8.set_xticklabels(politicians, rotation=45, ha='right')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # 9. Sector analysis
    ax9 = plt.subplot(5, 2, 9)
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
    
    ax9.bar(x - width/2, sector_avg_scores, width, label='Avg Arbitrage Score', alpha=0.8)
    ax9.bar(x + width/2, sector_avg_returns, width, label='Avg Return During Arbitrage', alpha=0.8)
    
    ax9.set_title('Sector Performance Analysis', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Sector', fontsize=12)
    ax9.set_ylabel('Score/Return', fontsize=12)
    ax9.set_xticks(x)
    ax9.set_xticklabels(sectors, rotation=45, ha='right')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)

    # 10. Neural Network Confidence Analysis
    ax10 = plt.subplot(5, 2, 10)
    if 'nn_probabilities' in arbitrage_results:
        for i, stock in enumerate(stocks):
            nn_probs = arbitrage_results['nn_probabilities'][stock]
            ax10.hist(nn_probs, bins=30, alpha=0.6, label=stock, color=colors[i], density=True)
        
        ax10.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Confidence Threshold')
        ax10.set_title('Neural Network Confidence Distribution', fontsize=14, fontweight='bold')
        ax10.set_xlabel('Arbitrage Probability', fontsize=12)
        ax10.set_ylabel('Density', fontsize=12)
        ax10.legend(fontsize=10)
        ax10.grid(True, alpha=0.3)

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
    """Print detailed arbitrage analysis with politician context and neural network comparison"""
    print("=== POLITICAL ARBITRAGE OPPORTUNITY ANALYSIS ===")
    print("=== INCLUDING NEURAL NETWORK PATTERN RECOGNITION ===")

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

    # Neural Network Analysis
    if 'nn_predictions' in arbitrage_results:
        print("\n" + "="*60)
        print("NEURAL NETWORK ARBITRAGE DETECTION RESULTS")
        print("="*60)
        
        for stock in stocks:
            nn_predictions = arbitrage_results['nn_predictions'][stock]
            nn_probabilities = arbitrage_results['nn_probabilities'][stock]
            
            # Get politician metadata
            politician_info = ""
            if stock in arbitrage_results['politician_metadata']:
                metadata = arbitrage_results['politician_metadata'][stock]
                politician_info = f" ({metadata['politician']}, {metadata['party']}, {metadata['sector']})"
            
            nn_arbitrage_count = nn_predictions.sum()
            avg_nn_probability = nn_probabilities.mean()
            max_nn_probability = nn_probabilities.max()
            
            print(f"\n{stock}{politician_info} - Neural Network Analysis:")
            print(f"  NN Detected Arbitrage Periods: {nn_arbitrage_count}")
            print(f"  Average NN Probability: {avg_nn_probability:.4f}")
            print(f"  Max NN Probability: {max_nn_probability:.4f}")
            
            # Show top NN probability periods
            top_nn_periods = nn_probabilities.nlargest(3)
            print(f"  Top 3 NN High-Probability Periods:")
            for date, prob in top_nn_periods.items():
                return_val = arbitrage_results['returns'].loc[date, stock]
                volume_val = arbitrage_results['volume_data'].loc[date, stock]
                print(f"    {date.strftime('%Y-%m-%d')}: Probability={prob:.4f}, Return={return_val:.4f}, Volume={volume_val:,.0f}")

    # Combined Analysis
    if 'combined_arbitrage_periods' in arbitrage_results:
        print("\n" + "="*60)
        print("COMBINED DETECTION RESULTS (Rule-Based OR Neural Network)")
        print("="*60)
        
        for stock in stocks:
            combined_periods = arbitrage_results['combined_arbitrage_periods'][stock]
            rule_based_periods = arbitrage_results['high_arbitrage_periods'][stock]
            nn_periods = arbitrage_results['nn_predictions'][stock] if 'nn_predictions' in arbitrage_results else pd.Series([False] * len(combined_periods))
            
            # Get politician metadata
            politician_info = ""
            if stock in arbitrage_results['politician_metadata']:
                metadata = arbitrage_results['politician_metadata'][stock]
                politician_info = f" ({metadata['politician']}, {metadata['party']}, {metadata['sector']})"
            
            combined_count = combined_periods.sum()
            rule_based_count = rule_based_periods.sum()
            nn_count = nn_periods.sum()
            
            print(f"\n{stock}{politician_info} - Combined Detection:")
            print(f"  Rule-Based Only: {rule_based_count}")
            print(f"  Neural Network Only: {nn_count}")
            print(f"  Combined Total: {combined_count}")
            print(f"  Additional Periods Detected by NN: {combined_count - rule_based_count}")
            
            if combined_count > 0:
                combined_dates = combined_periods[combined_periods].index
                combined_returns = arbitrage_results['returns'].loc[combined_dates, stock]
                print(f"  Average Return (Combined): {combined_returns.mean():.4f}")
                print(f"  Volatility (Combined): {combined_returns.std():.4f}")

    # Model Performance Summary
    if 'mlp_model' in arbitrage_results:
        print("\n" + "="*60)
        print("NEURAL NETWORK MODEL PERFORMANCE")
        print("="*60)
        
        mlp_model = arbitrage_results['mlp_model']
        print(f"  Model Type: Multi-Layer Perceptron Classifier")
        print(f"  Hidden Layers: {mlp_model.hidden_layer_sizes}")
        print(f"  Activation Function: {mlp_model.activation}")
        print(f"  Solver: {mlp_model.solver}")
        print(f"  Max Iterations: {mlp_model.max_iter}")
        
        # Feature importance analysis
        print(f"\n  Feature Analysis:")
        print(f"    - Price Features: Last 5 periods of returns")
        print(f"    - Volume Features: Volume ratio")
        print(f"    - Political Features: Surge indicators")
        print(f"    - Metadata Features: Party, Sector, Volume category")
        print(f"    - Total Features: 10 features per prediction")

def main():
    """Main function to run the political arbitrage detection"""
    # Define stocks with political context
    stocks = ['META', 'NFLX', 'NOW', 'FLL']
    
    # Define politician metadata from your notebook
    politician_metadata = {
        'META': {'politician': 'John McGuire', 'party': 'Republican', 'sector': 'Technology', 'volume': '1K-5K'},
        'NFLX': {'politician': 'John McGuire', 'party': 'Republican', 'sector': 'Technology', 'volume': '1K-5K'},
        'NOW': {'politician': 'John McGuire', 'party': 'Republican', 'sector': 'Technology', 'volume': '1K-5K'},
        'FLL': {'politician': 'Susie Lee', 'party': 'Democratic', 'sector': 'Gaming', 'volume': '50K-100K'}
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