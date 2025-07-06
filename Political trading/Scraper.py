from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime 

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
    exit()  

# 4. Scrape first 100 rows of trades
df_rows = []
rows = table.find_all("div", class_="trade-table-row", limit=100)

for row in rows:
    try:
        rep = row.find("div", class_="rep-info").get_text(strip=True)
        issuer = row.find("div", class_="issuer").get_text(strip=True)
        ticker = row.find("div", class_="ticker").get_text(strip=True)

        date = row.find("div", class_="trade-date").get_text(strip=True)# read as date
        # Convert date string to datetime object
        date = datetime.datetime.strptime(date, "%m/%d/%Y").date() 
        price = row.find("div", class_="price").get_text(strip=True)

        df_rows.append({
            "Representative": rep,
            "Ticker": ticker,
            "Date": date,
            "Issuer": issuer,
            "Price": price
        })
    except Exception as e:
        print("Skipping row due to error:", e)

df = pd.DataFrame(df_rows)

driver.quit()

# 5. Dump into CSV
#df = pd.DataFrame(data)
#df.to_csv("capitol_trades.csv", index=False)
#print("Saved to capitol_trades.csv")
