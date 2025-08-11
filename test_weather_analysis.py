#!/usr/bin/env python3
"""
Test script for weather data analysis
Tests if the current implementation can handle weather CSV analysis
"""
import requests
import json
import pandas as pd
import io

def test_weather_analysis():
    """Test the API with weather data analysis"""
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Expected fields for weather analysis
    expected_fields = [
        'average_temp_c',
        'max_precip_date',
        'min_temp_c', 
        'temp_precip_correlation',
        'average_precip_mm',
        'temp_line_chart',
        'precip_histogram'
    ]
    
    try:
        # Prepare files to upload
        files = []
        
        # Add questions.txt
        with open('weather-questions.txt', 'r') as f:
            questions_content = f.read()
        files.append(('files', ('questions.txt', questions_content, 'text/plain')))
        
        # Add weather CSV
        with open('sample-weather.csv', 'rb') as f:
            csv_content = f.read()
        files.append(('files', ('sample-weather.csv', csv_content, 'text/csv')))
        
        print("Sending weather analysis request to API...")
        response = requests.post(url, files=files, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Weather Analysis Results ===")
            
            found_fields = []
            missing_fields = []
            
            for field in expected_fields:
                if field in result:
                    value = result[field]
                    found_fields.append(field)
                    if isinstance(value, str) and value.startswith('data:image'):
                        print(f"✓ {field}: [base64 image - {len(value)} chars]")
                    else:
                        print(f"✓ {field}: {value}")
                else:
                    missing_fields.append(field)
                    print(f"✗ {field}: MISSING")
            
            # Show any unexpected fields
            unexpected_fields = set(result.keys()) - set(expected_fields)
            if unexpected_fields:
                print(f"\n⚠ Unexpected fields found: {list(unexpected_fields)}")
                for field in unexpected_fields:
                    print(f"  {field}: {result[field]}")
            
            print(f"\n=== Summary ===")
            print(f"✓ Found: {len(found_fields)}/{len(expected_fields)} expected fields")
            print(f"✗ Missing: {missing_fields}")
            
            return result, found_fields, missing_fields
            
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None, [], expected_fields
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:8000")
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        print("Make sure sample-weather.csv and weather-questions.txt are in the current directory")
    except Exception as e:
        print(f"Error: {e}")
    
    return None, [], []

def validate_weather_results(result):
    """Validate the weather analysis results against expected values"""
    if not result:
        print("No results to validate")
        return
        
    print("\n=== Data Validation ===")
    
    # Load the actual data to calculate expected values
    try:
        df = pd.read_csv('sample-weather.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        expected_avg_temp = df['temperature_c'].mean()
        expected_min_temp = df['temperature_c'].min()
        expected_avg_precip = df['precip_mm'].mean()
        expected_max_precip_date = df.loc[df['precip_mm'].idxmax(), 'date'].strftime('%Y-%m-%d')
        expected_correlation = df['temperature_c'].corr(df['precip_mm'])
        
        print(f"Expected average temp: {expected_avg_temp:.2f}°C")
        print(f"Expected min temp: {expected_min_temp}°C")
        print(f"Expected average precip: {expected_avg_precip:.2f}mm")
        print(f"Expected max precip date: {expected_max_precip_date}")
        print(f"Expected correlation: {expected_correlation:.4f}")
        
        # Validate results
        if 'average_temp_c' in result:
            actual = result['average_temp_c']
            if abs(actual - expected_avg_temp) < 0.01:
                print(f"✓ Average temperature correct: {actual}°C")
            else:
                print(f"✗ Average temperature wrong: got {actual}, expected {expected_avg_temp:.2f}")
        
        if 'min_temp_c' in result:
            actual = result['min_temp_c']
            if actual == expected_min_temp:
                print(f"✓ Min temperature correct: {actual}°C")
            else:
                print(f"✗ Min temperature wrong: got {actual}, expected {expected_min_temp}")
        
        if 'average_precip_mm' in result:
            actual = result['average_precip_mm']
            if abs(actual - expected_avg_precip) < 0.01:
                print(f"✓ Average precipitation correct: {actual}mm")
            else:
                print(f"✗ Average precipitation wrong: got {actual}, expected {expected_avg_precip:.2f}")
        
        if 'max_precip_date' in result:
            actual = result['max_precip_date']
            if actual == expected_max_precip_date:
                print(f"✓ Max precip date correct: {actual}")
            else:
                print(f"✗ Max precip date wrong: got {actual}, expected {expected_max_precip_date}")
        
        if 'temp_precip_correlation' in result:
            actual = result['temp_precip_correlation']
            if abs(actual - expected_correlation) < 0.01:
                print(f"✓ Correlation correct: {actual:.4f}")
            else:
                print(f"✗ Correlation wrong: got {actual:.4f}, expected {expected_correlation:.4f}")
        
        # Check visualizations
        if 'temp_line_chart' in result:
            if result['temp_line_chart'] and result['temp_line_chart'].startswith('data:image'):
                print("✓ Temperature line chart generated")
            else:
                print("✗ Temperature line chart missing or invalid")
        
        if 'precip_histogram' in result:
            if result['precip_histogram'] and result['precip_histogram'].startswith('data:image'):
                print("✓ Precipitation histogram generated")
            else:
                print("✗ Precipitation histogram missing or invalid")
                
    except Exception as e:
        print(f"Error validating results: {e}")

def analyze_implementation_gaps(missing_fields):
    """Analyze what's missing in the current implementation"""
    print("\n=== Implementation Gap Analysis ===")
    
    if not missing_fields:
        print("✅ All expected fields found! Current implementation handles weather data well.")
        return
    
    print(f"Missing {len(missing_fields)} fields:")
    
    for field in missing_fields:
        if field == 'average_temp_c':
            print(f"  ✗ {field}: Need to calculate mean of temperature column")
        elif field == 'min_temp_c':
            print(f"  ✗ {field}: Need to find minimum temperature value")
        elif field == 'max_precip_date':
            print(f"  ✗ {field}: Need to find date with highest precipitation")
        elif field == 'temp_precip_correlation':
            print(f"  ✗ {field}: Need to calculate correlation between temperature and precipitation")
        elif field == 'average_precip_mm':
            print(f"  ✗ {field}: Need to calculate mean of precipitation column")
        elif field == 'temp_line_chart':
            print(f"  ✗ {field}: Need to create line chart of temperature over time with red line")
        elif field == 'precip_histogram':
            print(f"  ✗ {field}: Need to create histogram of precipitation with orange bars")
    
    print("\nRecommended improvements:")
    print("1. Enhance weather data detection in routing logic")
    print("2. Add weather-specific analysis functions")
    print("3. Improve time series visualization capabilities")
    print("4. Add support for weather-specific metrics")

def display_test_info():
    """Display information about the weather analysis test"""
    print("=== Weather Data Analysis Test ===")
    print("Testing weather data with:")
    print("  - 10 days of data (2024-01-01 to 2024-01-10)")
    print("  - Temperature in Celsius (2°C to 8°C)")
    print("  - Precipitation in millimeters (0mm to 5mm)")
    print()
    print("Expected analysis:")
    print("  1. Average temperature calculation")
    print("  2. Find date with maximum precipitation")
    print("  3. Minimum temperature identification")  
    print("  4. Temperature-precipitation correlation")
    print("  5. Average precipitation calculation")
    print("  6. Temperature line chart (red line)")
    print("  7. Precipitation histogram (orange bars)")
    print()
    print("This will test if our current generic analysis can handle weather data")
    print("or if we need weather-specific analysis functions.")

if __name__ == "__main__":
    display_test_info()
    print("\n" + "="*70 + "\n")
    
    # Run the weather analysis test
    result, found_fields, missing_fields = test_weather_analysis()
    
    # Validate results if we got any
    if result:
        validate_weather_results(result)
    
    # Analyze implementation gaps
    analyze_implementation_gaps(missing_fields)
    
    print(f"\n{'='*70}")
    print("Test completed. Check results above to see if current implementation")
    print("can handle weather data analysis or needs improvements.")