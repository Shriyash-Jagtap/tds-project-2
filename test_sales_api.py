#!/usr/bin/env python3
"""
Simple test script to test the API with sales data
"""
import requests
import json

def test_sales_api():
    """Test the API with sales data"""
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    try:
        # Prepare files to upload
        files = []
        
        # Add questions.txt
        with open('test-questions.txt', 'r') as f:
            questions_content = f.read()
        files.append(('files', ('questions.txt', questions_content, 'text/plain')))
        
        # Add sales CSV
        with open('sample-sales.csv', 'rb') as f:
            csv_content = f.read()
        files.append(('files', ('sample-sales.csv', csv_content, 'text/csv')))
        
        print("Sending request to API...")
        response = requests.post(url, files=files, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== API Response Analysis ===")
            for key, value in result.items():
                if isinstance(value, str) and value.startswith('data:image'):
                    print(f"{key}: [base64 image data - {len(value)} chars]")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sales_api()