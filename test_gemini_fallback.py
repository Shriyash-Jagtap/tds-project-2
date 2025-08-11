#!/usr/bin/env python3
"""
Test script for Gemini LLM fallback functionality
Tests scenarios that should trigger the LLM instead of specific analysis code
"""
import requests
import json
import time

def test_gemini_fallback():
    """Test various scenarios that should trigger Gemini LLM fallback"""
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    test_cases = [
        {
            "name": "General Data Science Question",
            "questions": """What is machine learning and how does it work?
            
            Explain the difference between supervised and unsupervised learning.
            Give me examples of each type.""",
            "expected_behavior": "Should provide educational explanation about ML concepts"
        },
        {
            "name": "Business Analysis Request", 
            "questions": """I need help with my startup business plan. 
            
            What are the key metrics I should track for a SaaS company?
            How do I calculate customer lifetime value?
            What's a good customer acquisition cost?""",
            "expected_behavior": "Should provide business advice and formulas"
        },
        {
            "name": "Random CSV Analysis",
            "csv_data": """name,age,city
John,25,New York
Jane,30,San Francisco
Bob,35,Chicago
Alice,28,Boston""",
            "questions": """Analyze this employee data.
            
            Tell me interesting insights about this team.
            What patterns do you see?
            Any recommendations?""",
            "expected_behavior": "Should analyze the generic data and provide insights"
        },
        {
            "name": "Creative Writing Request",
            "questions": """Write a short story about a data scientist who discovers 
            something mysterious in a dataset.
            
            Make it exciting and include some technical details.""",
            "expected_behavior": "Should write a creative story"
        },
        {
            "name": "Programming Help",
            "questions": """Help me debug this Python code:
            
            def calculate_average(numbers):
                return sum(numbers) / len(numbers)
            
            result = calculate_average([])
            print(result)
            
            Why does this give an error? How do I fix it?""",
            "expected_behavior": "Should identify the division by zero error and suggest fixes"
        },
        {
            "name": "Statistical Question",
            "questions": """Explain what p-values mean in statistical testing.
            
            When is a p-value significant?
            What are Type I and Type II errors?
            Give me a practical example.""",
            "expected_behavior": "Should explain statistical concepts clearly"
        }
    ]
    
    print("=== Testing Gemini LLM Fallback Scenarios ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. Testing: {test_case['name']}")
        print(f"Expected: {test_case['expected_behavior']}")
        print("-" * 60)
        
        try:
            # Prepare files
            files = [('files', ('questions.txt', test_case['questions'], 'text/plain'))]
            
            # Add CSV data if provided
            if 'csv_data' in test_case:
                files.append(('files', ('data.csv', test_case['csv_data'], 'text/csv')))
            
            # Send request
            print("Sending request to API...")
            start_time = time.time()
            response = requests.post(url, files=files, timeout=30)
            end_time = time.time()
            
            print(f"Response Time: {end_time - start_time:.2f} seconds")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Check if it's structured JSON (from specific analysis) or text (from LLM)
                    if isinstance(result, dict) and len(result) == 1 and isinstance(list(result.values())[0], str):
                        # Likely LLM response wrapped in JSON
                        print("✓ LLM Fallback Triggered")
                        print("Response Type: Text-based LLM response")
                        response_text = list(result.values())[0]
                        print(f"Response Length: {len(response_text)} characters")
                        print("Sample Response:")
                        print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
                        
                    elif isinstance(result, str):
                        # Direct text response
                        print("✓ LLM Fallback Triggered")
                        print("Response Type: Direct text response")
                        print(f"Response Length: {len(result)} characters")
                        print("Sample Response:")
                        print(result[:300] + "..." if len(result) > 300 else result)
                        
                    elif isinstance(result, dict) and any(key in ['rows', 'columns', 'edge_count', 'total_sales'] for key in result.keys()):
                        # Structured analysis response
                        print("✗ Specific Analysis Triggered (Not LLM fallback)")
                        print("Response Type: Structured data analysis")
                        print(f"Keys found: {list(result.keys())}")
                        
                    else:
                        print("? Unknown Response Type")
                        print(f"Result type: {type(result)}")
                        print(f"Result: {result}")
                        
                except json.JSONDecodeError:
                    # Raw text response
                    print("✓ LLM Fallback Triggered")
                    print("Response Type: Raw text (not JSON)")
                    print(f"Response Length: {len(response.text)} characters")
                    print("Sample Response:")
                    print(response.text[:300] + "..." if len(response.text) > 300 else response.text)
                    
            else:
                print(f"✗ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("✗ Request timed out (LLM might be taking too long)")
        except requests.exceptions.ConnectionError:
            print("✗ Could not connect to API")
        except Exception as e:
            print(f"✗ Error: {e}")
            
        print("\n" + "="*80 + "\n")
        
        # Small delay between requests
        time.sleep(1)

def test_specific_vs_fallback():
    """Test that specific analysis still works vs fallback"""
    
    print("=== Comparing Specific Analysis vs LLM Fallback ===\n")
    
    url = "http://localhost:8000/api/"
    
    # Test 1: Should trigger specific network analysis
    print("1. Testing Network Analysis (Should NOT use LLM)")
    network_csv = "source,target\nAlice,Bob\nBob,Carol\nCarol,David"
    network_questions = "How many edges are in this network? What is the average degree?"
    
    files = [
        ('files', ('questions.txt', network_questions, 'text/plain')),
        ('files', ('edges.csv', network_csv, 'text/csv'))
    ]
    
    try:
        response = requests.post(url, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'edge_count' in result or 'average_degree' in result:
                print("✓ Specific Network Analysis Used")
                print(f"Keys: {list(result.keys())}")
            else:
                print("✗ LLM Fallback Used Instead")
                print(f"Response: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Test 2: Should trigger LLM fallback
    print("2. Testing Generic Question (Should use LLM)")
    generic_questions = "What's the capital of France and why is Paris important?"
    
    files = [('files', ('questions.txt', generic_questions, 'text/plain'))]
    
    try:
        response = requests.post(url, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, str) or (isinstance(result, dict) and len(result) == 1):
                print("✓ LLM Fallback Used")
                print("Response is text-based")
            else:
                print("✗ Specific Analysis Used Instead")
                print(f"Keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    except Exception as e:
        print(f"✗ Error: {e}")

def display_test_info():
    """Display information about the LLM fallback test"""
    print("=== Gemini LLM Fallback Test ===")
    print("This test will:")
    print("1. Send questions that DON'T match specific analysis patterns")
    print("2. Verify that the API falls back to Gemini LLM")
    print("3. Test the quality of LLM responses for various scenarios")
    print("4. Compare specific analysis vs LLM fallback")
    print()
    print("Test scenarios:")
    print("  - General ML/AI questions")
    print("  - Business advice requests") 
    print("  - Creative writing")
    print("  - Programming help")
    print("  - Statistical explanations")
    print("  - Generic CSV analysis")
    print()
    print("Expected behavior:")
    print("  ✓ Should use Gemini LLM for text-based responses")
    print("  ✓ Should provide helpful, contextual answers")
    print("  ✗ Should NOT return structured data analysis")

if __name__ == "__main__":
    display_test_info()
    print("\n" + "="*80 + "\n")
    
    # Test LLM fallback scenarios
    test_gemini_fallback()
    
    # Compare specific vs fallback
    test_specific_vs_fallback()