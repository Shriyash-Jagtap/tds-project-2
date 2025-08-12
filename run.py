#!/usr/bin/env python3
"""Runner script for the data analysis API - used by test framework"""

import sys
import requests
import json
import tempfile
import os

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No API URL provided"}))
        sys.exit(1)
    
    api_url = sys.argv[1]
    
    # Read questions from stdin or temp file
    questions_content = ""
    csv_files = {}
    
    # Look for questions.txt and CSV files in current directory
    if os.path.exists('questions.txt'):
        with open('questions.txt', 'r') as f:
            questions_content = f.read()
    
    # Look for CSV files
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            with open(file, 'rb') as f:
                csv_files[file] = f.read()
    
    # If no questions found, read from stdin
    if not questions_content:
        questions_content = sys.stdin.read()
    
    # Prepare files for upload
    files = []
    
    # Add questions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name
    
    try:
        with open(questions_file, 'rb') as qf:
            files.append(('files', ('questions.txt', qf, 'text/plain')))
            
            # Add CSV files
            temp_csv_files = []
            for filename, content in csv_files.items():
                temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False)
                temp_file.write(content)
                temp_file.close()
                temp_csv_files.append((temp_file.name, filename))
            
            # Open all CSV files and add to request
            csv_handles = []
            for temp_path, orig_name in temp_csv_files:
                handle = open(temp_path, 'rb')
                csv_handles.append(handle)
                files.append(('files', (orig_name, handle, 'text/csv')))
            
            # Send request to API
            try:
                response = requests.post(api_url, files=files, timeout=30)
                
                if response.status_code == 200:
                    # Return the JSON response
                    result = response.json()
                    print(json.dumps(result))
                else:
                    print(json.dumps({"error": f"API returned status {response.status_code}"}))
                    
            except requests.exceptions.RequestException as e:
                print(json.dumps({"error": str(e)}))
            finally:
                # Close all file handles
                for handle in csv_handles:
                    handle.close()
                    
    finally:
        # Clean up temp files
        os.unlink(questions_file)
        for temp_path, _ in temp_csv_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    main()
