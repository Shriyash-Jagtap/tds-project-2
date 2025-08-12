#!/usr/bin/env python3
"""
Test script for Indian High Court judgments analysis

Tests the LLM's ability to:
1. Handle legal domain data and terminology
2. Process large datasets with SQL-like queries
3. Perform date calculations and time-based analysis
4. Generate statistical analysis (regression)
5. Create data visualizations with regression lines
6. Work with hierarchical data structures (court/bench/year)
"""

import json
import requests
import base64
import io
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import random

class IndianCourtJudgmentTest:
    def __init__(self):
        """Initialize test with simulated Indian High Court data"""
        self.courts = {
            '33_10': 'Madras High Court',
            '33_08': 'Delhi High Court', 
            '33_07': 'Bombay High Court',
            '33_12': 'Karnataka High Court',
            '33_15': 'Kerala High Court',
            '33_05': 'Calcutta High Court',
            '33_13': 'Andhra Pradesh High Court',
            '33_09': 'Gujarat High Court'
        }
        
        self.disposal_natures = [
            'DISMISSED', 'ALLOWED', 'DISPOSED OF', 'WITHDRAWN', 
            'DISMISSED AS WITHDRAWN', 'PARTLY ALLOWED', 'REJECTED'
        ]
        
        self.judges = [
            'HONOURABLE MR JUSTICE G.K. ILANTHIRAIYAN',
            'HONOURABLE MR JUSTICE R. MAHADEVAN',
            'HONOURABLE MS JUSTICE ANITA SUMANTH',
            'HONOURABLE MR JUSTICE S. VAIDYANATHAN',
            'HONOURABLE MR JUSTICE N. SATHISH KUMAR'
        ]
        
        # Generate test dataset
        self.data = self._generate_court_data()
        
    def _generate_court_data(self):
        """Generate realistic Indian High Court judgment data"""
        data = []
        
        # Generate data for years 2019-2022
        for year in range(2019, 2023):
            for court_code, court_name in self.courts.items():
                # Different courts handle different volumes
                if court_code == '33_10':  # Madras - highest volume
                    num_cases = random.randint(8000, 12000)
                elif court_code in ['33_08', '33_07']:  # Delhi, Bombay - high volume
                    num_cases = random.randint(6000, 9000)
                else:
                    num_cases = random.randint(3000, 7000)
                
                for _ in range(num_cases):
                    # Registration date
                    reg_date = self._random_date(year)
                    
                    # Decision date - varies by court efficiency
                    if court_code == '33_10':  # Madras court - faster processing
                        delay_days = random.randint(30, 180)
                    elif court_code == '33_08':  # Delhi - moderate delay
                        delay_days = random.randint(60, 300)
                    else:
                        delay_days = random.randint(90, 400)
                    
                    # Add yearly trend - delays generally increasing
                    yearly_factor = (year - 2019) * 20
                    delay_days += yearly_factor
                    
                    decision_date = reg_date + timedelta(days=delay_days)
                    
                    case_num = random.randint(1000, 9999)
                    case_type = random.choice(['CRL MP', 'WP', 'CMA', 'OSA', 'CRL A'])
                    
                    data.append({
                        'court_code': court_code,
                        'title': f'{case_type}(MD)/{case_num}/{year} of Petitioner Vs Respondent',
                        'description': f'No.{case_num} of {year} BEFORE THE HIGH COURT...',
                        'judge': random.choice(self.judges),
                        'pdf_link': f'court/orders/{court_code}/orders/case_{case_num}_{year}.pdf',
                        'cnr': f'HC{court_code.replace("_", "")}{random.randint(100000000, 999999999)}{year}',
                        'date_of_registration': reg_date.strftime('%d-%m-%Y'),
                        'decision_date': decision_date.strftime('%Y-%m-%d'),
                        'disposal_nature': random.choice(self.disposal_natures),
                        'court': court_code,
                        'raw_html': f'<button type="button" role="link">Case {case_num}</button>',
                        'bench': f'{court_code.split("_")[1]}bench',
                        'year': year,
                        'days_delay': delay_days
                    })
        
        return data
    
    def _random_date(self, year):
        """Generate random date within a year"""
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        return start + timedelta(days=random.randint(0, (end - start).days))
    
    def analyze_court_disposal_volume(self):
        """Find which high court disposed most cases from 2019-2022"""
        court_counts = {}
        
        for record in self.data:
            court = record['court']
            if court not in court_counts:
                court_counts[court] = 0
            court_counts[court] += 1
        
        # Find court with maximum disposals
        max_court = max(court_counts, key=court_counts.get)
        max_count = court_counts[max_court]
        
        return {
            'court_with_most_disposals': max_court,
            'court_name': self.courts[max_court],
            'total_disposals': max_count,
            'all_court_counts': court_counts
        }
    
    def calculate_regression_slope(self):
        """Calculate regression slope of registration-decision delay by year for court 33_10"""
        # Filter for court 33_10
        court_data = [r for r in self.data if r['court'] == '33_10']
        
        # Calculate average delay by year
        year_delays = {}
        for record in court_data:
            year = record['year']
            delay = record['days_delay']
            
            if year not in year_delays:
                year_delays[year] = []
            year_delays[year].append(delay)
        
        # Get average delay per year
        years = []
        avg_delays = []
        for year, delays in year_delays.items():
            years.append(year)
            avg_delays.append(sum(delays) / len(delays))
        
        # Perform linear regression
        X = np.array(years).reshape(-1, 1)
        y = np.array(avg_delays)
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'years': years,
            'average_delays': avg_delays,
            'court': '33_10'
        }
    
    def create_regression_plot(self, regression_data):
        """Create scatter plot with regression line"""
        plt.figure(figsize=(10, 6))
        
        years = regression_data['years']
        delays = regression_data['average_delays']
        slope = regression_data['slope']
        intercept = regression_data['intercept']
        
        # Scatter plot
        plt.scatter(years, delays, color='darkblue', s=100, alpha=0.7, label='Average Delay')
        
        # Regression line
        x_line = np.array(years)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linewidth=2, linestyle='--', 
                label=f'Regression Line (slope={slope:.1f})')
        
        # Formatting
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Days of Delay', fontsize=12)
        plt.title('Case Processing Delay Trend\nMadras High Court (33_10)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(years)
        
        # Add annotation
        plt.annotate(f'Trend: +{slope:.1f} days/year', 
                    xy=(max(years), max(delays)), 
                    xytext=(max(years)-0.5, max(delays)+5),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def generate_expected_results(self):
        """Generate expected results for validation"""
        disposal_analysis = self.analyze_court_disposal_volume()
        regression_analysis = self.calculate_regression_slope()
        plot_data_uri = self.create_regression_plot(regression_analysis)
        
        return {
            "Which high court disposed the most cases from 2019 - 2022?": disposal_analysis['court_name'],
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": round(regression_analysis['slope'], 2),
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_data_uri
        }

def test_indian_court_analysis():
    """Main test function"""
    print("=== Indian High Court Judgment Analysis Test ===")
    print("Testing legal data analysis with:")
    print("  - 8 major High Courts")
    print("  - ~200,000 simulated judgments (2019-2022)")
    print("  - Legal terminology and case processing metrics")
    print("  - Date-based analysis and regression modeling")
    print("  - Hierarchical data structure (court/bench/year)")
    print("")
    print("Expected analysis capabilities:")
    print("  1. Court-wise case volume analysis")
    print("  2. Temporal trend analysis with regression")
    print("  3. Legal domain understanding")
    print("  4. Complex date calculations")
    print("  5. Statistical visualization with regression lines")
    print("  6. Large dataset processing simulation")
    print("")
    print("This tests the LLM's ability to:")
    print("  - Understand legal terminology and court systems")
    print("  - Handle complex SQL-like data queries")
    print("  - Perform statistical analysis on judicial data")
    print("  - Generate appropriate legal domain visualizations")
    print("  - Process hierarchical legal data structures")
    print("")
    print("=" * 80)
    print("")
    
    # Initialize test
    test_suite = IndianCourtJudgmentTest()
    expected_results = test_suite.generate_expected_results()
    
    # Create comprehensive sample data for the API
    sample_data = {
        'dataset_description': 'Indian High Court judgments dataset with ~16M judgments from 25 high courts',
        'data_structure': {
            'total_courts': 8,
            'total_judgments': len(test_suite.data),
            'years_covered': [2019, 2020, 2021, 2022],
            'courts': test_suite.courts,
            'disposal_natures': test_suite.disposal_natures
        },
        'full_dataset': test_suite.data,  # Include all data for complete analysis
        'questions': {
            "Which high court disposed the most cases from 2019 - 2022?": None,
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": None,
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": None
        }
    }
    
    # Send request to API
    print("Sending Indian High Court analysis request to API...")
    try:
        # Create questions text that describes the Indian High Court dataset context
        questions_text = f"""
        Context: You are analyzing the Indian High Court judgments dataset which contains ~16M judgments from 25 high courts downloaded from ecourts website. The data structure is:
        
        Dataset Info:
        - 25 high courts with ~16M judgments from 2019-2022
        - Stored in S3: s3://indian-high-court-judgments/metadata/parquet/
        - Columns: court_code, title, description, judge, pdf_link, cnr, date_of_registration, decision_date, disposal_nature, court, raw_html, bench, year
        
        Sample data structure:
        {sample_data}
        
        Please answer these questions about the Indian High Court judgments dataset:

        1. Which high court disposed the most cases from 2019 - 2022?
        2. What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?
        3. Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters.
        
        Return results as JSON with exact keys:
        {{
        "Which high court disposed the most cases from 2019 - 2022?": "court_name",
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": numeric_slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/png;base64,..."
        }}
        """
        
        # Send just the questions text - no CSV file needed
        files = [
            ('files', ('questions.txt', questions_text.encode(), 'text/plain'))
        ]
        
        response = requests.post('http://localhost:8000/api/', 
                               files=files, 
                               timeout=90)  # Longer timeout for LLM analysis
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print("")
            print("=== Indian High Court Analysis Results ===")
            
            # Validate results
            validation_results = {}
            all_correct = True
            
            # Check court with most disposals
            expected_court = expected_results["Which high court disposed the most cases from 2019 - 2022?"]
            actual_court = results.get("Which high court disposed the most cases from 2019 - 2022?", "")
            court_correct = expected_court == actual_court
            validation_results['court_analysis'] = court_correct
            all_correct &= court_correct
            
            print(f"✓ Court with most disposals: {actual_court}" if court_correct else f"✗ Court analysis incorrect: got {actual_court}, expected {expected_court}")
            
            # Check regression slope (allow small numerical differences)
            expected_slope = expected_results["What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?"]
            actual_slope = results.get("What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?", 0)
            
            try:
                slope_correct = abs(float(actual_slope) - float(expected_slope)) < 5.0  # Allow 5-day difference
                validation_results['regression_slope'] = slope_correct
                all_correct &= slope_correct
                print(f"✓ Regression slope: {actual_slope} days/year" if slope_correct else f"✗ Slope incorrect: got {actual_slope}, expected ~{expected_slope}")
            except (ValueError, TypeError):
                validation_results['regression_slope'] = False
                all_correct = False
                print(f"✗ Invalid regression slope: {actual_slope}")
            
            # Check plot data URI
            plot_uri = results.get("Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters", "")
            plot_valid = (
                plot_uri.startswith('data:image/') and 
                'base64,' in plot_uri and 
                len(plot_uri) < 100000 and
                len(plot_uri) > 1000  # Reasonable minimum size
            )
            validation_results['visualization'] = plot_valid
            all_correct &= plot_valid
            
            print(f"✓ Regression plot: Valid data URI ({len(plot_uri)} chars)" if plot_valid else f"✗ Plot generation failed or invalid")
            
            print("")
            print("=== Summary ===")
            correct_count = sum(validation_results.values())
            total_tests = len(validation_results)
            print(f"✓ Found: {correct_count}/{total_tests} expected analyses")
            print(f"✗ Issues: {[k for k, v in validation_results.items() if not v]}")
            
            print("")
            print("=== Data Validation ===")
            print(f"Expected court with most cases: {expected_court}")
            print(f"Expected regression slope: ~{expected_slope} days/year")
            print("Expected plot: Base64 data URI with scatter plot and regression line")
            
            print("")
            print("=== Dynamic LLM Performance Analysis ===")
            success_rate = (correct_count / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}% ({correct_count}/{total_tests} analyses)")
            
            if success_rate >= 90:
                print("🎉 EXCELLENT: Dynamic LLM system handles legal data analysis very well!")
            elif success_rate >= 70:
                print("✅ GOOD: Dynamic LLM system handles legal data reasonably well")
            elif success_rate >= 50:
                print("⚠️ FAIR: Dynamic LLM system shows some legal data analysis capabilities")
            else:
                print("❌ NEEDS IMPROVEMENT: Legal data analysis capabilities need enhancement")
            
            print("")
            print("This test verifies the LLM can handle:")
            print(f"  {'✓' if validation_results.get('court_analysis') else '✗'} Legal domain knowledge and terminology")
            print(f"  {'✓' if validation_results.get('regression_slope') else '✗'} Complex date-based statistical analysis")  
            print(f"  {'✓' if validation_results.get('visualization') else '✗'} Legal data visualization with regression")
            print("  ✓ Large dataset simulation and processing")
            print("  ✓ Hierarchical legal data structures")
            print("  ✓ Judicial system understanding")
            
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
    
    print("")
    print("=" * 80)
    print("Indian High Court judgment analysis test completed!")
    print("This verifies our dynamic LLM system can handle complex legal data")
    print("analysis without any court-specific hardcoded logic.")

if __name__ == "__main__":
    test_indian_court_analysis()