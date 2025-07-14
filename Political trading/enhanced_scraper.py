import os
import sys
import time
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin

class CapitolTradesScraper:
    def __init__(self, headless=True):
        """Initialize the scraper with Chrome options"""
        self.base_url = "https://www.capitoltrades.com"
        self.driver = None
        self.setup_driver(headless)
    
    def setup_driver(self, headless=True):
        """Setup Chrome driver with appropriate options"""
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            sys.exit(1)
    
    def parse_amount(self, amount_str):
        """Parse amount string to numeric range"""
        if not amount_str:
            return None, None
        
        # Remove dollar signs and commas
        amount_str = amount_str.replace('$', '').replace(',', '')
        
        def convert_k_to_number(text):
            """Convert K notation to number (e.g., '1K' -> 1000)"""
            text = text.strip().upper()
            if text.endswith('K'):
                return float(text[:-1]) * 1000
            elif text.endswith('M'):
                return float(text[:-1]) * 1000000
            else:
                return float(text)
        
        # Handle ranges like "1K–15K" or "1K - 15K"
        if '–' in amount_str or '-' in amount_str:
            # Use – (em dash) or - (hyphen)
            separator = '–' if '–' in amount_str else '-'
            parts = amount_str.split(separator)
            if len(parts) == 2:
                try:
                    min_amount = convert_k_to_number(parts[0])
                    max_amount = convert_k_to_number(parts[1])
                    return min_amount, max_amount
                except ValueError:
                    return None, None
        
        # Handle single values
        try:
            amount = convert_k_to_number(amount_str)
            return amount, amount
        except ValueError:
            return None, None
    
    def extract_trade_data(self, row):
        """Extract comprehensive trade data from a single row"""
        try:
            trade_data = {}
            
            # Trade action (Buy/Sell) - updated for new structure
            action_element = row.find("span", class_="q-field tx-type")
            if action_element:
                action_text = action_element.get_text(strip=True).lower()
                if 'buy' in action_text or 'purchase' in action_text:
                    trade_data['action'] = 'BUY'
                elif 'sell' in action_text or 'sale' in action_text:
                    trade_data['action'] = 'SELL'
                else:
                    trade_data['action'] = action_text.upper()
            
            # Trade date - updated for new structure
            date_element = row.find("span", class_="time")
            if date_element:
                date_text = date_element.get_text(strip=True)
                try:
                    if "Today" in date_text:
                        trade_data['trade_date'] = datetime.date.today()
                        trade_data['days_ago'] = 0
                    elif "Yesterday" in date_text:
                        trade_data['trade_date'] = datetime.date.today() - datetime.timedelta(days=1)
                        trade_data['days_ago'] = 1
                    else:
                        # Try to parse other date formats
                        trade_data['trade_date'] = None
                        trade_data['days_ago'] = None
                except ValueError:
                    trade_data['trade_date'] = None
                    trade_data['days_ago'] = None
            
            # Stock information - updated for new structure
            issuer_element = row.find("h3", class_="q-fieldset issuer-name")
            if issuer_element:
                trade_data['issuer'] = issuer_element.get_text(strip=True)
            
            ticker_element = row.find("span", class_="q-field issuer-ticker")
            if ticker_element:
                ticker_text = ticker_element.get_text(strip=True)
                # Remove :US suffix if present
                trade_data['ticker'] = ticker_text.replace(':US', '')
            
            # Representative information - updated for new structure
            rep_element = row.find("h2", class_="politician-name")
            if rep_element:
                trade_data['representative'] = rep_element.get_text(strip=True)
            
            # Party information - updated for new structure
            party_element = row.find("span", class_="q-field party")
            if party_element:
                party_text = party_element.get_text(strip=True)
                if 'Republican' in party_text:
                    trade_data['party'] = 'Republican'
                elif 'Democrat' in party_text:
                    trade_data['party'] = 'Democrat'
                else:
                    trade_data['party'] = 'Other'
            
            # Position information - updated for new structure
            chamber_element = row.find("span", class_="q-field chamber")
            if chamber_element:
                chamber_text = chamber_element.get_text(strip=True)
                if 'House' in chamber_text:
                    trade_data['position'] = 'House'
                elif 'Senate' in chamber_text:
                    trade_data['position'] = 'Senate'
                else:
                    trade_data['position'] = 'Unknown'
            
            # State information - updated for new structure
            state_element = row.find("span", class_="q-field us-state-compact")
            if state_element:
                trade_data['state'] = state_element.get_text(strip=True)
            
            # Amount/Volume - updated for new structure
            amount_element = row.find("span", class_="q-field trade-size")
            if amount_element:
                amount_text = amount_element.get_text(strip=True)
                min_amount, max_amount = self.parse_amount(amount_text)
                trade_data['amount_min'] = min_amount
                trade_data['amount_max'] = max_amount
                trade_data['amount_range'] = amount_text
            
            return trade_data
        
        except Exception as e:
            print(f"Error extracting trade data: {e}")
            return None
    
    def scrape_trades(self, max_trades=500): 
        """Scrape trade data from Capitol Trades"""
        try:
            print(f"Navigating to {self.base_url}")
            self.driver.get(self.base_url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Try to handle any popups or cookie notices
            try:
                close_button = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='close-button']")
                if close_button:
                    close_button.click()
                    time.sleep(2)
            except:
                pass
            
            # Get page source and parse
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find trade table (updated for current website structure)
            trade_table = soup.find("article", class_="q-data-list")
            if not trade_table:
                print("No q-data-list found. Searching for alternative selectors...")
                # Try alternative selectors
                trade_table = soup.find("div", class_="trade-table") or soup.find("div", class_="trades-table")
                if not trade_table:
                    print("No table found. Dumping page structure for analysis...")
                    with open("page_structure.html", "w") as f:
                        f.write(str(soup.prettify()))
                    return []
            
            # Extract all trade rows (updated for current structure)
            rows = trade_table.find_all("a", class_="link-element")
            
            if not rows:
                print("No link-element rows found. Trying alternative selectors...")
                rows = trade_table.find_all("div", class_="trade-row") or trade_table.find_all("div", class_="trade-table-row")
            
            if not rows:
                print("No rows found. Saving page for manual inspection...")
                with open("debug_page.html", "w") as f:
                    f.write(str(soup.prettify()))
                return []
            
            print(f"Found {len(rows)} trade rows")
            
            trades = []
            for i, row in enumerate(rows[:max_trades]):
                if i % 50 == 0:
                    print(f"Processing row {i+1}/{min(len(rows), max_trades)}")
                
                trade_data = self.extract_trade_data(row)
                if trade_data:
                    trades.append(trade_data)
                    
                # Small delay to avoid overwhelming the site
                if i % 10 == 0:
                    time.sleep(0.5)
            
            return trades
            
        except Exception as e:
            print(f"Error scraping trades: {e}")
            return []
        
        #add parameter to scrape trades of highest volume
    
    def analyze_trades(self, trades):
        """Analyze trade data for investment insights"""
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        
        analysis = {
            'total_trades': len(trades),
            'recent_activity': len(df[df['days_ago'] <= 30]) if 'days_ago' in df.columns else 0,
            'buy_sell_ratio': {},
            'most_active_stocks': {},
            'party_breakdown': {},
            'position_breakdown': {},
            'recent_big_trades': []
        }
        
        # Buy/Sell analysis
        if 'action' in df.columns:
            action_counts = df['action'].value_counts()
            analysis['buy_sell_ratio'] = action_counts.to_dict()
        
        # Most active stocks
        if 'ticker' in df.columns:
            ticker_counts = df['ticker'].value_counts().head(10)
            analysis['most_active_stocks'] = ticker_counts.to_dict()
        
        # Party breakdown
        if 'party' in df.columns:
            party_counts = df['party'].value_counts()
            analysis['party_breakdown'] = party_counts.to_dict()
        
        # Position breakdown
        if 'position' in df.columns:
            position_counts = df['position'].value_counts()
            analysis['position_breakdown'] = position_counts.to_dict()
        
        # Recent big trades (over $50k)
        if 'amount_min' in df.columns and 'days_ago' in df.columns:
            recent_big = df[(df['amount_min'] > 50000) & (df['days_ago'] <= 30)]
            if not recent_big.empty:
                analysis['recent_big_trades'] = recent_big.head(10).to_dict('records')
        
        return analysis
    
    def export_data(self, trades, filename='enhanced_capitol_trades.csv'):
        """Export trade data to CSV"""
        if not trades:
            print("No trades to export")
            return
        
        df = pd.DataFrame(trades)
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports', exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/capitol_trades_{timestamp}.csv"
        
        # Sort by date (most recent first)
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date', ascending=False)
        
        df.to_csv(filename, index=False)
        print(f"Exported {len(trades)} trades to {filename}")
        
        # Also create a summary file
        summary_filename = f"exports/trade_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("CAPITOL TRADES ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            analysis = self.analyze_trades(trades)
            
            f.write(f"Total Trades: {analysis['total_trades']}\n")
            f.write(f"Recent Activity (30 days): {analysis['recent_activity']}\n\n")
            
            f.write("BUY/SELL BREAKDOWN:\n")
            for action, count in analysis['buy_sell_ratio'].items():
                f.write(f"  {action}: {count}\n")
            
            f.write("\nMOST ACTIVE STOCKS:\n")
            for ticker, count in analysis['most_active_stocks'].items():
                f.write(f"  {ticker}: {count} trades\n")
            
            f.write("\nPARTY BREAKDOWN:\n")
            for party, count in analysis['party_breakdown'].items():
                f.write(f"  {party}: {count}\n")
        
        print(f"Summary saved to {summary_filename}")
    
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()

def main():
    """Main scraping function"""
    print("Starting Capitol Trades Enhanced Scraper...")
    
    scraper = CapitolTradesScraper(headless=True)
    
    try:
        # Scrape trades
        trades = scraper.scrape_trades(max_trades=500)
        
        if trades:
            print(f"Successfully scraped {len(trades)} trades")
            
            # Export data
            scraper.export_data(trades)
            
            # Print analysis
            analysis = scraper.analyze_trades(trades)
            print("\n" + "="*50)
            print("QUICK ANALYSIS:")
            print(f"Total Trades: {analysis['total_trades']}")
            print(f"Recent Activity (30 days): {analysis['recent_activity']}")
            print(f"Buy/Sell Ratio: {analysis['buy_sell_ratio']}")
            print(f"Most Active Stocks: {list(analysis['most_active_stocks'].keys())[:5]}")
            print("="*50)
            
        else:
            print("No trades found. Check the website structure.")
            
    except Exception as e:
        print(f"Error during scraping: {e}")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 