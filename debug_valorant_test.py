#!/usr/bin/env python3
"""Debug test to see what the API returns for Valorant analysis"""

import json
import requests

def debug_valorant_api():
    """Simple debug test"""
    print("=== Debug Valorant API Response ===")
    
    questions_text = """
    Please analyze the Valorant tournament VCT 2025 Americas League Stage 2 from https://liquipedia.net/valorant/VCT/2025/Americas_League/Stage_2

    Answer these questions:
    1. What countries are 100 Thieves players from?
    2. How many observers - Replay operators are there?
    3. Full name of the Producer?
    
    Return as JSON format.
    """
    
    files = [('files', ('questions.txt', questions_text.encode(), 'text/plain'))]
    
    try:
        response = requests.post('http://localhost:8000/api/', files=files, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"\nResponse Type: {type(result)}")
                print(f"Response Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                print(f"\nFull Response:")
                print(json.dumps(result, indent=2, default=str))
                
                # Test parsing JSON from markdown
                if isinstance(result, str):
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        print(f"\nExtracted JSON:")
                        print(json_str)
                        try:
                            parsed_json = json.loads(json_str)
                            print(f"\nParsed JSON Keys: {list(parsed_json.keys())}")
                        except Exception as e:
                            print(f"JSON parse error: {e}")
            except json.JSONDecodeError:
                print("\nRaw Response (not JSON):")
                print(response.text[:1000])
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_valorant_api()