from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime
import re 

def scrape_capitol_trades(max_volume_threshold=None, max_rows=500):
    """
    Scrape capitol trades with optional volume filtering
    
    Args:
        max_volume_threshold: Maximum volume threshold for filtering stocks
        max_rows: Maximum number of rows to scrape (default 500)
    """
    # 1. Set up Selenium headless Chrome
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    service = Service('/usr/local/bin/chromedriver') 
    driver = webdriver.Chrome(service=Service(), options=options)
    driver.get("https://www.capitoltrades.com/")

    time.sleep(7)  # Allow time for dynamic content to load

    # 3. Extract page HTML and parse
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    rows = soup.find_all("div", class_="trade-table-row")

    # set up HTML table scrape 
    table = soup.find("div", class_="trade-table")
    if not table:
        print("No trade table found on the page.")
        driver.quit()
        return pd.DataFrame()

    # 4. Scrape trades with increased limit
    df_rows = []
    rows = table.find_all("div", class_="trade-table-row", limit=max_rows)

    def parse_volume(volume_str):
        """Parse volume string and convert to numerical value"""
        if not volume_str or volume_str == "--":
            return 0
        
        volume_str = volume_str.replace("$", "").replace(",", "").strip()
        
        # Handle K, M, B suffixes
        if volume_str.endswith("K"):
            return float(volume_str[:-1]) * 1000
        elif volume_str.endswith("M"):
            return float(volume_str[:-1]) * 1000000
        elif volume_str.endswith("B"):
            return float(volume_str[:-1]) * 1000000000
        else:
            try:
                return float(volume_str)
            except:
                return 0

    for row in rows:
        try:
            rep = row.find("div", class_="rep-info").get_text(strip=True)
            issuer = row.find("div", class_="issuer").get_text(strip=True)
            ticker = row.find("div", class_="ticker").get_text(strip=True)

            date = row.find("div", class_="trade-date").get_text(strip=True)
            # Convert date string to datetime object
            date = datetime.datetime.strptime(date, "%m/%d/%Y").date() 
            price = row.find("div", class_="price").get_text(strip=True)
            
            # Extract volume data
            volume_element = row.find("div", class_="volume") or row.find("div", class_="amount")
            volume_str = volume_element.get_text(strip=True) if volume_element else "0"
            volume = parse_volume(volume_str)
            
            # Apply volume filter if specified
            if max_volume_threshold is not None and volume > max_volume_threshold:
                continue

            df_rows.append({
                "Representative": rep,
                "Ticker": ticker,
                "Date": date,
                "Issuer": issuer,
                "Price": price,
                "Volume": volume,
                "Volume_Raw": volume_str
            })
        except Exception as e:
            print("Skipping row due to error:", e)

    df = pd.DataFrame(df_rows)
    
    # Sort by volume in descending order to get highest volume stocks first
    if not df.empty and 'Volume' in df.columns:
        df = df.sort_values('Volume', ascending=False)
    
    driver.quit()
    
    return df

# Example usage functions
def get_high_volume_stocks(volume_threshold=1000000, max_rows=1000):
    """Get stocks with volume above threshold"""
    return scrape_capitol_trades(max_volume_threshold=None, max_rows=max_rows)

def get_top_volume_stocks(top_n=100, max_rows=1000):
    """Get top N stocks by volume"""
    df = scrape_capitol_trades(max_volume_threshold=None, max_rows=max_rows)
    return df.head(top_n) if not df.empty else df

# Main execution
if __name__ == "__main__":
    # Scrape with volume filtering for maximum traded stocks
    df = scrape_capitol_trades(max_volume_threshold=None, max_rows=1000)
    
    print(f"Scraped {len(df)} trades")
    if not df.empty:
        print(f"Volume range: ${df['Volume'].min():,.0f} - ${df['Volume'].max():,.0f}")
        print("\nTop 10 highest volume trades:")
        print(df.head(10)[['Ticker', 'Representative', 'Volume', 'Volume_Raw', 'Date']])
    
    # Save to CSV
    df.to_csv("capitol_trades_with_volume.csv", index=False)
    print("Saved to capitol_trades_with_volume.csv")
