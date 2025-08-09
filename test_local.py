import pandas as pd
import numpy as np

# Test scraping
try:
    dfs = pd.read_html("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
    df = dfs[0]
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for Rank and Peak columns
    for col in df.columns:
        if 'rank' in str(col).lower():
            print(f"\nFound Rank column: {col}")
            print(df[col].head())
        if 'peak' in str(col).lower():
            print(f"\nFound Peak column: {col}")
            print(df[col].head())
            
except Exception as e:
    print(f"Error: {e}")