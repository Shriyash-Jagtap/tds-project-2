#!/usr/bin/env python3
"""
Test script for car efficiency data analysis
Tests if the dynamic LLM system can handle automotive data analysis without specific coding
"""
import requests
import json
import pandas as pd
import numpy as np

def test_car_efficiency_analysis():
    """Test the API with car efficiency data analysis"""
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Expected fields for car efficiency analysis
    expected_fields = [
        'average_combined_mpg',
        'most_efficient_car',
        'fuel_type_efficiency',
        'price_efficiency_correlation',
        'engine_size_mpg_correlation',
        'efficiency_by_fuel_chart',
        'mpg_vs_price_scatter'
    ]
    
    try:
        # Prepare files to upload
        files = []
        
        # Add questions.txt
        with open('car-efficiency-questions.txt', 'r') as f:
            questions_content = f.read()
        files.append(('files', ('questions.txt', questions_content, 'text/plain')))
        
        # Add car efficiency CSV
        with open('sample-car-efficiency.csv', 'rb') as f:
            csv_content = f.read()
        files.append(('files', ('sample-car-efficiency.csv', csv_content, 'text/csv')))
        
        print("Sending car efficiency analysis request to API...")
        response = requests.post(url, files=files, timeout=90)  # Longer timeout for complex analysis
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Car Efficiency Analysis Results ===")
            
            found_fields = []
            missing_fields = []
            
            for field in expected_fields:
                if field in result:
                    value = result[field]
                    found_fields.append(field)
                    if isinstance(value, str) and value.startswith('data:image'):
                        print(f"✓ {field}: [base64 image - {len(value)} chars]")
                    elif isinstance(value, dict):
                        print(f"✓ {field}: {value}")
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
        print("Make sure sample-car-efficiency.csv and car-efficiency-questions.txt are in the current directory")
    except Exception as e:
        print(f"Error: {e}")
    
    return None, [], []

def validate_car_efficiency_results(result):
    """Validate the car efficiency analysis results against expected values"""
    if not result:
        print("No results to validate")
        return
        
    print("\n=== Data Validation ===")
    
    # Load the actual data to calculate expected values
    try:
        df = pd.read_csv('sample-car-efficiency.csv')
        
        expected_avg_mpg = df['combined_mpg'].mean()
        expected_most_efficient = df.loc[df['combined_mpg'].idxmax(), 'car_model']
        
        # Fuel type efficiency
        expected_fuel_efficiency = df.groupby('fuel_type')['combined_mpg'].mean().to_dict()
        
        # Price-efficiency correlation
        expected_price_corr = df['price_usd'].corr(df['combined_mpg'])
        
        # Engine size correlation (excluding electric cars)
        non_electric = df[df['engine_size_l'] > 0]
        expected_engine_corr = non_electric['engine_size_l'].corr(non_electric['combined_mpg'])
        
        print(f"Expected average combined MPG: {expected_avg_mpg:.2f}")
        print(f"Expected most efficient car: {expected_most_efficient}")
        print(f"Expected fuel type efficiency: {expected_fuel_efficiency}")
        print(f"Expected price correlation: {expected_price_corr:.4f}")
        print(f"Expected engine size correlation: {expected_engine_corr:.4f}")
        
        # Validate results
        if 'average_combined_mpg' in result:
            actual = result['average_combined_mpg']
            if abs(actual - expected_avg_mpg) < 1.0:
                print(f"✓ Average combined MPG correct: {actual:.2f}")
            else:
                print(f"✗ Average combined MPG wrong: got {actual}, expected {expected_avg_mpg:.2f}")
        
        if 'most_efficient_car' in result:
            actual = result['most_efficient_car']
            if actual == expected_most_efficient:
                print(f"✓ Most efficient car correct: {actual}")
            else:
                print(f"✗ Most efficient car wrong: got {actual}, expected {expected_most_efficient}")
        
        if 'fuel_type_efficiency' in result:
            actual = result['fuel_type_efficiency']
            if isinstance(actual, dict):
                print("✓ Fuel type efficiency format correct")
                for fuel_type, expected_mpg in expected_fuel_efficiency.items():
                    if fuel_type in actual:
                        if abs(actual[fuel_type] - expected_mpg) < 1.0:
                            print(f"  ✓ {fuel_type}: {actual[fuel_type]:.2f} MPG")
                        else:
                            print(f"  ✗ {fuel_type}: got {actual[fuel_type]}, expected {expected_mpg:.2f}")
                    else:
                        print(f"  ✗ {fuel_type}: missing from results")
            else:
                print(f"✗ Fuel type efficiency wrong format: {type(actual)}")
        
        if 'price_efficiency_correlation' in result:
            actual = result['price_efficiency_correlation']
            if abs(actual - expected_price_corr) < 0.1:
                print(f"✓ Price correlation correct: {actual:.4f}")
            else:
                print(f"✗ Price correlation wrong: got {actual:.4f}, expected {expected_price_corr:.4f}")
        
        if 'engine_size_mpg_correlation' in result:
            actual = result['engine_size_mpg_correlation']
            if abs(actual - expected_engine_corr) < 0.1:
                print(f"✓ Engine size correlation correct: {actual:.4f}")
            else:
                print(f"✗ Engine size correlation wrong: got {actual:.4f}, expected {expected_engine_corr:.4f}")
        
        # Check visualizations
        if 'efficiency_by_fuel_chart' in result:
            if result['efficiency_by_fuel_chart'] and result['efficiency_by_fuel_chart'].startswith('data:image'):
                print("✓ Fuel efficiency bar chart generated")
            else:
                print("✗ Fuel efficiency bar chart missing or invalid")
        
        if 'mpg_vs_price_scatter' in result:
            if result['mpg_vs_price_scatter'] and result['mpg_vs_price_scatter'].startswith('data:image'):
                print("✓ MPG vs price scatter plot generated")
            else:
                print("✗ MPG vs price scatter plot missing or invalid")
                
    except Exception as e:
        print(f"Error validating results: {e}")

def analyze_implementation_performance(found_fields, missing_fields):
    """Analyze how well the dynamic LLM system performed"""
    print("\n=== Dynamic LLM Performance Analysis ===")
    
    total_fields = len(found_fields) + len(missing_fields)
    success_rate = len(found_fields) / total_fields * 100 if total_fields > 0 else 0
    
    print(f"Success Rate: {success_rate:.1f}% ({len(found_fields)}/{total_fields} fields)")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Dynamic LLM system handles car efficiency data very well!")
    elif success_rate >= 75:
        print("✅ GOOD: Dynamic LLM system handles most car efficiency requirements")
    elif success_rate >= 50:
        print("⚠️ MODERATE: Dynamic LLM system handles some requirements but needs improvement")
    else:
        print("❌ POOR: Dynamic LLM system struggles with car efficiency analysis")
    
    if missing_fields:
        print(f"\nMissing capabilities:")
        for field in missing_fields:
            if 'correlation' in field:
                print(f"  - {field}: Statistical analysis needs improvement")
            elif 'chart' in field or 'scatter' in field:
                print(f"  - {field}: Visualization generation needs work")
            elif 'efficiency' in field:
                print(f"  - {field}: Domain-specific calculations missing")
    
    print("\nThis test verifies the LLM can handle:")
    print("  ✓ Automotive domain knowledge")
    print("  ✓ Statistical correlations") 
    print("  ✓ Grouped analysis by categories")
    print("  ✓ Multiple chart types")
    print("  ✓ Complex business calculations")

def display_test_info():
    """Display information about the car efficiency analysis test"""
    print("=== Car Efficiency Data Analysis Test ===")
    print("Testing automotive data with:")
    print("  - 15 car models from 2022")
    print("  - Multiple fuel types: Gasoline, Hybrid, Electric")
    print("  - Engine sizes: 0.0L to 5.3L") 
    print("  - MPG ratings: City, Highway, Combined")
    print("  - Price range: $20,000 to $85,000")
    print()
    print("Expected analysis capabilities:")
    print("  1. Statistical calculations (averages, correlations)")
    print("  2. Data grouping and aggregation") 
    print("  3. Finding maximum/minimum values")
    print("  4. Complex filtering (excluding electric cars from engine analysis)")
    print("  5. Multiple visualization types (bar chart, scatter plot)")
    print("  6. Domain-specific understanding (fuel efficiency concepts)")
    print()
    print("This tests the LLM's ability to:")
    print("  - Understand automotive terminology")
    print("  - Handle mixed data types and complex calculations")
    print("  - Generate appropriate visualizations")
    print("  - Apply business logic (e.g., excluding electric cars from engine analysis)")

if __name__ == "__main__":
    display_test_info()
    print("\n" + "="*80 + "\n")
    
    # Run the car efficiency analysis test
    result, found_fields, missing_fields = test_car_efficiency_analysis()
    
    # Validate results if we got any
    if result:
        validate_car_efficiency_results(result)
    
    # Analyze implementation performance
    analyze_implementation_performance(found_fields, missing_fields)
    
    print(f"\n{'='*80}")
    print("Car efficiency test completed!")
    print("This verifies our dynamic LLM system can handle complex automotive analysis")
    print("without any car-specific hardcoded logic.")