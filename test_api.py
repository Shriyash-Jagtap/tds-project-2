import requests
import json
import sys

def test_local_api():
    """Test the API locally"""
    url = "http://localhost:8000/api/"
    
    with open("questions.txt", "rb") as f:
        files = [("files", ("questions.txt", f, "text/plain"))]
        
        try:
            response = requests.post(url, files=files, timeout=180)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\nResponse:")
                print(json.dumps(result, indent=2))
                
                if isinstance(result, list) and len(result) == 4:
                    print("\n✓ Response has correct structure (4-element array)")
                    print(f"✓ Answer 1 (2bn movies before 2000): {result[0]}")
                    print(f"✓ Answer 2 (earliest 1.5bn film): {result[1]}")
                    print(f"✓ Answer 3 (correlation): {result[2]}")
                    print(f"✓ Answer 4 (plot): {'data:image' in str(result[3])}")
                else:
                    print("\n✗ Response structure is incorrect")
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print("Request timed out (180s)")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_local_api()