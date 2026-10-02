import os
import requests
import pandas as pd
from datetime import datetime

# 1. Configuration parameters for last Friday (September 4, 2026)
target_symbol = "NIFTY26SEP24500CE"  
start_dt = int(datetime(2026, 9, 4, 9, 15).timestamp())
end_dt = int(datetime(2026, 9, 4, 15, 30).timestamp())

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://nseindia.com"
}

# 2. Start a continuous browser session to retain cookies
session = requests.Session()
session.get("https://nseindia.com", headers=headers)

try:
    # 3. VERIFY THIS LINE AFTER PASTING: Ensure it reads '.com/api/' explicitly
    search_url = f"https://nseindia.comapi/search/autocomplete?q={target_symbol}"
    search_data = session.get(search_url, headers=headers).json()
    
    token = None
    for entry in search_data.get('symbols', []):
        if entry.get('symbol') == target_symbol:
            token = entry.get('scripcode')
            break
            
    if not token:
        raise ValueError(f"Could not find a valid scripcode token matching {target_symbol}")
        
    print(f"Token found: {token}. Extracting 1-minute candle bars...")

    # 4. Request the historical 1-minute array database directly
    chart_url = f"https://nseindia.comapi/chart-databycharttype?index={token}&type=candles&start={start_dt}&end={end_dt}"
    chart_data = session.get(chart_url, headers=headers).json()
    
    # "grapthData" is the raw structural dictionary key on the live exchange endpoint
    candles = chart_data.get('grapthData', [])
    if candles:
        df = pd.DataFrame(candles, columns=['Timestamp_Epoch', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp_Epoch'], unit='s')
        df.drop(columns=['Timestamp_Epoch'], inplace=True)
        
        # Save straight to local CSV
        output_file = f"{target_symbol}_1min.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSuccess! File saved as: {os.path.abspath(output_file)}")
        print(df.head())
    else:
        print("No chart bars returned from the server query for this time range.")

except Exception as e:
    print(f"Scraper pipeline execution failed: {e}")
