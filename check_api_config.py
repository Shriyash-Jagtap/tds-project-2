#!/usr/bin/env python3
"""
Quick check to see if the Gemini API is properly configured
"""
import os
from dotenv import load_dotenv

def check_gemini_config():
    """Check if Gemini API is configured"""
    load_dotenv()
    
    api_key = os.getenv("AIPIPE_API_KEY", "")
    
    print("=== Gemini API Configuration Check ===")
    
    if api_key:
        print(f"✓ AIPIPE_API_KEY found (length: {len(api_key)} chars)")
        print(f"  Preview: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
        
        # Test the API directly
        try:
            import requests
            
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "google/gemini-2.0-flash-exp",
                "messages": [{"role": "user", "content": "Hello, this is a test. Respond with 'API working'."}],
                "temperature": 0.3,
                "max_tokens": 50,
                "stream": False
            }
            
            print("Testing API connection...")
            response = requests.post("https://aipipe.org/openrouter/v1/chat/completions", json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"✓ API connection successful!")
                    print(f"  Test response: {content}")
                else:
                    print(f"⚠ API responded but unexpected format: {result}")
            else:
                print(f"✗ API error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"✗ API test failed: {e}")
            
    else:
        print("✗ AIPIPE_API_KEY not found")
        print("  Set it in .env file: AIPIPE_API_KEY=your_key_here")
        print("  Or as environment variable")
        print("  Get key from: https://aipipe.org")
        
    print()
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ .env file exists")
        with open('.env', 'r') as f:
            content = f.read()
            if 'AIPIPE_API_KEY' in content:
                print("✓ .env file contains AIPIPE_API_KEY")
            else:
                print("✗ .env file missing AIPIPE_API_KEY")
    else:
        print("⚠ .env file not found")
        print("  Create one with: AIPIPE_API_KEY=your_key_here")

if __name__ == "__main__":
    check_gemini_config()