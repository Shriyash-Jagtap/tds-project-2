import pandas as pd
import requests
from bs4 import BeautifulSoup

def test_scraping():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    
    print("Testing Method 1: pandas read_html direct")
    try:
        tables = pd.read_html(url, header=0, attrs={'class': 'wikitable'})
        if tables:
            df = tables[0]
            print(f"✓ Success: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First row: {df.iloc[0].tolist()}")
            return df
        else:
            print("✗ No tables found")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTesting Method 2: requests + pandas")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        print(f"Response status: {response.status_code}")
        
        tables = pd.read_html(response.content)
        if tables:
            df = tables[0]
            print(f"✓ Success: {df.shape}")
            return df
        else:
            print("✗ No tables found")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTesting Method 3: BeautifulSoup manual")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        tables = soup.find_all('table', {'class': 'wikitable'})
        print(f"Found {len(tables)} wikitable elements")
        
        if tables:
            table = tables[0]
            rows = table.find_all('tr')
            print(f"Found {len(rows)} rows in first table")
            
            if rows:
                # Print first few rows
                for i, row in enumerate(rows[:3]):
                    cells = row.find_all(['td', 'th'])
                    cell_texts = [cell.text.strip() for cell in cells]
                    print(f"Row {i}: {cell_texts}")
                
                return True
        else:
            print("✗ No wikitable found")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    result = test_scraping()
    if result is not None:
        print(f"\n✓ At least one method worked!")
    else:
        print(f"\n✗ All methods failed")