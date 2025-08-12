#!/usr/bin/env python3
"""
Simple test for Indian High Court judgments analysis without requiring matplotlib
Tests the LLM's ability to handle legal data analysis via API calls only
"""

import json
import requests

def test_indian_court_simple():
    """Simple test for Indian High Court analysis"""
    print("=== Indian High Court Judgment Analysis (Simple Test) ===")
    print("Testing legal data analysis without local dependencies")
    print("This tests pure API functionality for legal domain analysis")
    print("")
    
    try:
        # Create comprehensive questions about Indian High Court data
        questions_text = """
        Context: You are analyzing the Indian High Court judgments dataset which contains ~16M judgments from 25 high courts downloaded from the ecourts website. The data has this structure:
        
        Dataset Information:
        - Source: https://judgments.ecourts.gov.in/
        - 25 high courts with ~16M judgments (2019-2022)
        - Stored in S3: s3://indian-high-court-judgments/metadata/parquet/
        - Structure: data/pdf/year=*/court=*/bench=*/judgment.pdf
        - Metadata: metadata/parquet/year=*/court=*/bench=*/metadata.parquet
        
        Columns:
        - court_code: Court identifier (e.g., 33~10)
        - title: Case title and parties
        - description: Case description
        - judge: Presiding judge(s) 
        - pdf_link: Link to judgment PDF
        - cnr: Case Number Register
        - date_of_registration: Registration date (DD-MM-YYYY format)
        - decision_date: Date of judgment (YYYY-MM-DD format)
        - disposal_nature: Case outcome (DISMISSED, ALLOWED, etc.)
        - court: Court name (33_10 = Madras High Court)
        - raw_html: Original HTML content
        - bench: Bench identifier  
        - year: Year partition
        
        Sample record:
        {
          "court_code": "33~10",
          "title": "CRL MP(MD)/4399/2023 of Vinoth Vs The Inspector of Police",
          "description": "No.4399 of 2023 BEFORE THE MADURAI BENCH OF MADRAS HIGH COURT...",
          "judge": "HONOURABLE MR JUSTICE G.K. ILANTHIRAIYAN",
          "pdf_link": "court/cnrorders/mdubench/orders/HCMD010287762023_1_2023-03-16.pdf",
          "cnr": "HCMD010287762023",
          "date_of_registration": "14-03-2023",
          "decision_date": "2023-03-16",
          "disposal_nature": "DISMISSED",
          "court": "33_10",
          "bench": "mdubench", 
          "year": 2023
        }
        
        Please analyze this dataset and answer the following questions:

        1. Which high court disposed the most cases from 2019 - 2022?
        2. What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?
        3. Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters.
        
        Return results as JSON with these exact keys:
        {
        "Which high court disposed the most cases from 2019 - 2022?": "court_name",
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": numeric_slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/png;base64,..."
        }
        """
        
        # Send just the questions - no data file needed
        files = [
            ('files', ('questions.txt', questions_text.encode(), 'text/plain'))
        ]
        
        print("Sending Indian High Court analysis request to API...")
        response = requests.post('http://localhost:8000/api/', 
                               files=files, 
                               timeout=120)  # Long timeout for LLM processing
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                results = response.json()
                print("")
                print("=== API Response Analysis ===")
                
                # Check what we got back
                if isinstance(results, dict):
                    print(f"✓ Received dictionary response with {len(results)} keys")
                    for key, value in results.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  - {key}: {type(value).__name__} ({len(value)} chars)")
                        else:
                            print(f"  - {key}: {value}")
                elif isinstance(results, str):
                    print(f"✓ Received string response ({len(results)} chars)")
                    if results.startswith('{') and results.endswith('}'):
                        print("  - Looks like JSON string - trying to parse...")
                        try:
                            parsed = json.loads(results)
                            results = parsed
                            print(f"  - Successfully parsed: {len(parsed)} keys")
                        except:
                            print("  - Failed to parse as JSON")
                else:
                    print(f"✓ Received {type(results).__name__} response")
                
                # Analyze the response for legal domain understanding
                success_indicators = []
                
                # Check for court analysis
                for key in results:
                    if "high court" in key.lower():
                        value = results[key]
                        if isinstance(value, str) and ("high court" in value.lower() or "court" in value.lower()):
                            success_indicators.append("Court identification")
                        break
                
                # Check for regression analysis
                for key in results:
                    if "regression" in key.lower() and "slope" in key.lower():
                        value = results[key]
                        if isinstance(value, (int, float)):
                            success_indicators.append("Regression analysis")
                        break
                
                # Check for visualization
                for key in results:
                    if "plot" in key.lower() or "scatter" in key.lower():
                        value = results[key]
                        if isinstance(value, str) and value.startswith("data:image"):
                            success_indicators.append("Data visualization")
                        break
                
                print("")
                print("=== Legal Domain Analysis Results ===")
                print(f"✓ API Success: {len(success_indicators)}/3 capabilities demonstrated")
                for indicator in success_indicators:
                    print(f"  ✓ {indicator}")
                
                missing = ["Court identification", "Regression analysis", "Data visualization"]
                for indicator in success_indicators:
                    if indicator in missing:
                        missing.remove(indicator)
                
                for missed in missing:
                    print(f"  ✗ {missed} - not detected")
                
                print("")
                if len(success_indicators) >= 2:
                    print("🎉 EXCELLENT: API demonstrates strong legal domain analysis!")
                elif len(success_indicators) >= 1:
                    print("✅ GOOD: API shows legal domain understanding")
                else:
                    print("⚠️ NEEDS IMPROVEMENT: Limited legal domain analysis detected")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw response: {response.text[:500]}...")
                return False
                
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        return False
    
    print("")
    print("=" * 80)
    print("Indian High Court simple analysis test completed!")

if __name__ == "__main__":
    test_indian_court_simple()