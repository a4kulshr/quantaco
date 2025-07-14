from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# 1. Set up Selenium headless Chrome
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
service = Service('/usr/local/bin/chromedriver') 
driver = webdriver.Chrome(service=Service(), options=options)
driver.get("https://www.capitoltrades.com/trades")

#insert a delay to allow page to load
WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, "site-main")))

# 3. Extract page HTML and parse
soup = BeautifulSoup(driver.page_source, 'html.parser')
rows = soup.find_all("div", class_="site-main")

# set up HTML table scrape 
table = soup.find("div", class_="trade-table")
if not table:
    print("No trade table found on the page.")
    driver.quit()
    exit()  

# 4. Scrape first 100 rows of trades
df_rows = []
rows = table.find_all("div", class_="site-main", limit=100)
#read by indexing by tr number
for row in rows:
    try:
        politician = row.find_all[0].get_text(strip=True)
        trade_issuer = row.find_all[1].get_text(strip=True)
        traded_at = row.find_all[3].get_text(strip=True)
        type = row.find_all[6].get_text(strip=True)
        size = row.find_all[7].get_text(strip=True)
        price = row.find_all[8].get_text(strip=True)
    except Exception as e:
        print(f"Error processing row: {e}")


df = pd.DataFrame(df_rows)

driver.quit()

# 5. Dump into CSV
#df = pd.DataFrame(data)
#df.to_csv("capitol_trades.csv", index=False)
#print("Saved to capitol_trades.csv")
#find.all (class_="trade-table")