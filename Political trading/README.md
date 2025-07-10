# 🏛️ Enhanced Capitol Trades Scraper

A comprehensive Python scraper for extracting congressional trading data from [Capitol Trades](https://www.capitoltrades.com/) with advanced analysis capabilities for investment decision-making.

## 🚀 Features

### Enhanced Data Extraction
- **Comprehensive Trade Data**: Representative name, ticker symbol, company name, trade amounts, dates
- **Political Context**: Party affiliation, position (House/Senate), state representation
- **Investment Analysis**: Volume analysis, recent activity tracking, big trades monitoring
- **Robust Parsing**: Handles current website structure with error handling and fallbacks

### Investment Intelligence
- **Buy/Sell Ratio Analysis**: Track market sentiment among politicians
- **Most Active Stocks**: Identify stocks with highest political interest
- **Big Trades Tracking**: Monitor significant trades (>$50K)
- **Recent Activity Focus**: Emphasis on trades within last 30 days
- **Party Breakdown**: Compare trading patterns by political affiliation

### Data Export & Analysis
- **CSV Export**: Timestamped data exports with comprehensive trade details
- **Analysis Summary**: Automated investment insights and trend reports
- **Debug Support**: Detailed error handling and troubleshooting capabilities

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Chrome browser installed
- ChromeDriver (handled automatically by webdriver-manager)

### Setup Instructions

1. **Clone or download the project files**:
```bash
# Ensure you have these files:
# - enhanced_scraper.py
# - run_scraper.py
# - requirements.txt
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify Chrome installation**:
The scraper uses Chrome WebDriver which will be automatically managed by the `webdriver-manager` package.

## 🔧 Usage

### Quick Start
```bash
# Run with default settings (500 trades, headless mode)
python run_scraper.py

# Run with custom settings
python run_scraper.py --max-trades 100 --visible
```

### Command Line Options
```bash
# Basic usage
python run_scraper.py

# Scrape specific number of trades
python run_scraper.py --max-trades 250

# Run in visible mode for debugging
python run_scraper.py --visible

# Custom output file
python run_scraper.py --output my_trades.csv

# Show help
python run_scraper.py --help
```

### Direct Python Usage
```python
from enhanced_scraper import CapitolTradesScraper

# Initialize scraper
scraper = CapitolTradesScraper(headless=True)

try:
    # Scrape trades
    trades = scraper.scrape_trades(max_trades=100)
    
    # Analyze and export
    if trades:
        scraper.export_data(trades)
        analysis = scraper.analyze_trades(trades)
        print(f"Scraped {len(trades)} trades")
    
finally:
    scraper.close()
```

## 📊 Output Format

### CSV Export Fields
The scraper exports comprehensive trade data with these fields:

| Field | Description | Example |
|-------|-------------|---------|
| `trade_date` | Date of the trade | 2025-07-09 |
| `days_ago` | Days since the trade | 0 |
| `issuer` | Company name | Broadcom Inc |
| `ticker` | Stock ticker symbol | AVGO |
| `representative` | Politician name | John Boozman |
| `action` | BUY or SELL | BUY |
| `party` | Political party | Republican |
| `position` | House or Senate | Senate |
| `state` | State represented | AR |
| `amount_min` | Minimum trade amount | 1000.0 |
| `amount_max` | Maximum trade amount | 15000.0 |
| `amount_range` | Original amount text | 1K–15K |

### Analysis Summary
Automatically generated summaries include:
- **Total trades** captured
- **Recent activity** (last 30 days)
- **Buy/sell ratio** breakdown
- **Most active stocks** by trade volume
- **Party breakdown** of trading activity
- **Recent big trades** (>$50K)

## 🎯 Investment Use Cases

### 1. Following Political Sentiment
```python
# Track what politicians are buying/selling
analysis = scraper.analyze_trades(trades)
hot_stocks = analysis['most_active_stocks']
print("Politicians are actively trading:", list(hot_stocks.keys())[:5])
```

### 2. Identifying Market Trends
```python
# Find stocks with recent heavy activity
recent_trades = [t for t in trades if t['days_ago'] <= 7]
print(f"Trades in last week: {len(recent_trades)}")
```

### 3. Party-Based Analysis
```python
# Compare trading patterns by political party
party_breakdown = analysis['party_breakdown']
print("Trading by party:", party_breakdown)
```

## 🔍 Example Output

### Console Output
```
============================================================
🏛️  CAPITOL TRADES ENHANCED SCRAPER 🏛️
============================================================
📊 Max trades: 100
🖥️  Headless: True
📁 Output: exports/capitol_trades.csv
============================================================

🔍 Starting scrape...
Navigating to https://www.capitoltrades.com
Found 5 trade rows
Processing row 1/5
✅ Successfully scraped 5 trades

==================================================
📊 INVESTMENT ANALYSIS:
==================================================
📈 Total Trades: 5
🔥 Recent Activity (30 days): 5

🎯 MOST ACTIVE STOCKS:
  TEAM: 2 trades
  AVGO: 1 trades
  AMZN: 1 trades
  APP: 1 trades
==================================================
✅ Scraping completed successfully!
📁 Check the 'exports/' directory for detailed data and analysis.
```

### Sample CSV Data
```csv
trade_date,days_ago,issuer,ticker,representative,amount_min,amount_max,amount_range
2025-07-09,0,Broadcom Inc,AVGO,John Boozman,1000.0,15000.0,1K–15K
2025-07-09,0,Amazon.com Inc,AMZN,Rob Bresnahan,1000.0,15000.0,1K–15K
2025-07-09,0,APPLOVIN CORP,APP,Rob Bresnahan,1000.0,15000.0,1K–15K
2025-07-09,0,Atlassian Corp PLC,TEAM,Rob Bresnahan,1000.0,15000.0,1K–15K
```

## 🛠️ Configuration

### Scraper Settings
```python
# Initialize with custom settings
scraper = CapitolTradesScraper(
    headless=True,  # Run in headless mode
)

# Scrape with custom parameters
trades = scraper.scrape_trades(
    max_trades=500,  # Number of trades to scrape
)
```

### Output Customization
The scraper automatically creates timestamped files in the `exports/` directory:
- `capitol_trades_YYYYMMDD_HHMMSS.csv` - Main data file
- `trade_summary_YYYYMMDD_HHMMSS.txt` - Analysis summary

## 🐛 Troubleshooting

### Common Issues

1. **No trades found**
   - Website structure may have changed
   - Try running with `--visible` to see browser activity
   - Check if Capitol Trades website is accessible

2. **ChromeDriver issues**
   ```bash
   # The webdriver-manager should handle this automatically
   # If issues persist, ensure Chrome is updated
   ```

3. **Rate limiting**
   - The scraper includes delays to avoid overwhelming the server
   - If you encounter issues, try reducing `max_trades`

4. **Incomplete data**
   - Some fields may be missing depending on website changes
   - Check debug output files for details

### Debug Mode
```bash
# Run with visible browser to see what's happening
python run_scraper.py --visible --max-trades 10
```

### Debug Files
When issues occur, the scraper creates debug files:
- `page_structure.html` - Full page HTML for analysis
- `debug_page.html` - Specific section causing issues

## 📁 File Structure

```
Political trading/
├── enhanced_scraper.py     # Main scraper class
├── run_scraper.py          # Easy-to-use runner script
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── exports/               # Generated export files
│   ├── capitol_trades_*.csv
│   └── trade_summary_*.txt
└── debug_files/           # Debug output (if needed)
```

## 🔗 Dependencies

The scraper uses these key libraries:
- **selenium**: Web browser automation
- **beautifulsoup4**: HTML parsing
- **pandas**: Data manipulation and export
- **webdriver-manager**: Automatic ChromeDriver management

## 📈 Performance

- **Speed**: Scrapes ~5 trades from homepage (limited by website)
- **Reliability**: Robust error handling and retries
- **Memory**: Minimal memory usage with efficient parsing
- **Rate Limiting**: Built-in delays to respect website servers

## 🔒 Legal & Ethics

This scraper is designed for:
- **Educational purposes** - Learning about web scraping and data analysis
- **Research activities** - Academic or personal research on political trading
- **Investment insights** - Understanding market trends from public data

Please ensure you:
- Respect Capitol Trades' terms of service
- Use reasonable request rates (built into the scraper)
- Attribute data sources appropriately
- Consider the legal implications of your use case

## 🚀 Future Enhancements

Potential improvements for the scraper:
- **Historical data**: Scrape older trades beyond the homepage
- **Real-time monitoring**: Continuous monitoring with notifications
- **Advanced analysis**: Statistical analysis and trend prediction
- **API integration**: Connect with trading APIs for alerts
- **Data visualization**: Charts and graphs for trade patterns

## 🤝 Contributing

To improve the scraper:
1. Test with different website conditions
2. Report bugs or website structure changes
3. Suggest new features or analysis capabilities
4. Optimize parsing performance

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run with `--visible` flag to debug
3. Examine debug files created by the scraper
4. Ensure all dependencies are properly installed

---

**Happy trading analysis! 📊🏛️**

*This tool provides valuable insights into congressional trading patterns for informed investment decisions.* 