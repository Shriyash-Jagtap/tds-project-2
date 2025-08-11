#!/usr/bin/env python3
"""
Test script for Titanic Wikipedia scraping
Tests the API's ability to scrape and analyze Wikipedia movie pages
"""
import requests
import json

def test_titanic_wikipedia():
    """Test the API with Titanic Wikipedia scraping"""
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    try:
        # Prepare files to upload
        files = []
        
        # Add questions.txt for Titanic
        with open('titanic-questions.txt', 'r') as f:
            questions_content = f.read()
        files.append(('files', ('questions.txt', questions_content, 'text/plain')))
        
        print("Sending Titanic Wikipedia scraping request to API...")
        response = requests.post(url, files=files, timeout=60)  # Longer timeout for web scraping
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Titanic Wikipedia Analysis Results ===")
            
            # Expected fields for Titanic analysis
            expected_fields = [
                'budget_millions',
                'worldwide_gross_millions', 
                'director',
                'runtime_minutes',
                'release_year',
                'production_companies',
                'main_cast_count'
            ]
            
            for field in expected_fields:
                if field in result:
                    value = result[field]
                    print(f"✓ {field}: {value}")
                else:
                    print(f"✗ {field}: MISSING")
            
            # Validation of expected Titanic data
            print("\n=== Data Validation ===")
            
            # Check budget (Titanic had a budget of ~$200 million)
            if 'budget_millions' in result:
                budget = result['budget_millions']
                if 150 <= budget <= 250:  # Allow some range for different sources
                    print(f"✓ Budget reasonable: ${budget} million")
                else:
                    print(f"⚠ Budget seems off: ${budget} million (expected ~$200M)")
            
            # Check gross (Titanic made over $2 billion worldwide)
            if 'worldwide_gross_millions' in result:
                gross = result['worldwide_gross_millions']
                if gross > 2000:  # Should be over $2 billion
                    print(f"✓ Box office correct: ${gross} million")
                else:
                    print(f"⚠ Box office seems low: ${gross} million (expected >$2B)")
            
            # Check director (James Cameron)
            if 'director' in result:
                director = result['director']
                if 'cameron' in director.lower():
                    print(f"✓ Director correct: {director}")
                else:
                    print(f"⚠ Director unexpected: {director} (expected James Cameron)")
            
            # Check release year (1997)
            if 'release_year' in result:
                year = result['release_year']
                if year == 1997:
                    print(f"✓ Release year correct: {year}")
                else:
                    print(f"⚠ Release year incorrect: {year} (expected 1997)")
            
            # Check runtime (should be around 195 minutes)
            if 'runtime_minutes' in result:
                runtime = result['runtime_minutes']
                if 190 <= runtime <= 200:
                    print(f"✓ Runtime reasonable: {runtime} minutes")
                else:
                    print(f"⚠ Runtime unexpected: {runtime} minutes (expected ~195)")
            
            # Check production companies
            if 'production_companies' in result:
                companies = result['production_companies']
                if isinstance(companies, list) and len(companies) > 0:
                    print(f"✓ Production companies found: {companies}")
                    # Check for known companies
                    companies_str = ' '.join(companies).lower()
                    if 'paramount' in companies_str or '20th century' in companies_str:
                        print("✓ Major studios identified correctly")
                else:
                    print(f"⚠ Production companies format issue: {companies}")
            
            # Check cast count
            if 'main_cast_count' in result:
                cast_count = result['main_cast_count']
                if cast_count >= 2:  # Should have at least DiCaprio and Winslet
                    print(f"✓ Cast count reasonable: {cast_count} main cast members")
                else:
                    print(f"⚠ Cast count seems low: {cast_count}")
            
            # Check for any errors
            if 'error' in result:
                print(f"✗ Error occurred: {result['error']}")
            
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:8000")
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        print("Make sure titanic-questions.txt is in the current directory")
    except Exception as e:
        print(f"Error: {e}")

def test_other_movies():
    """Test other movie Wikipedia scraping"""
    
    movies_to_test = [
        {
            "name": "Avatar (2009)",
            "questions": """Analyze the Wikipedia page for Avatar (2009 film).
            
            Return a JSON object with keys:
            - `budget_millions`: number
            - `worldwide_gross_millions`: number
            - `director`: string
            - `release_year`: number
            
            Answer:
            1. What was the budget of Avatar?
            2. What was the worldwide gross?
            3. Who directed Avatar?
            4. What year was Avatar released?""",
            "expected_year": 2009,
            "expected_director": "cameron"
        },
        {
            "name": "Avengers: Endgame",
            "questions": """Analyze the Wikipedia page for Avengers: Endgame.
            
            Return a JSON object with keys:
            - `budget_millions`: number
            - `worldwide_gross_millions`: number
            - `director`: string
            - `release_year`: number
            
            Answer:
            1. What was the budget of Avengers: Endgame?
            2. What was the worldwide gross?
            3. Who directed Avengers: Endgame?
            4. What year was Avengers: Endgame released?""",
            "expected_year": 2019,
            "expected_director": "russo"
        }
    ]
    
    url = "http://localhost:8000/api/"
    
    for movie in movies_to_test:
        print(f"\n{'='*20} Testing {movie['name']} {'='*20}")
        
        try:
            files = [('files', ('questions.txt', movie['questions'], 'text/plain'))]
            
            response = requests.post(url, files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Check year
                if 'release_year' in result:
                    if result['release_year'] == movie['expected_year']:
                        print(f"✓ {movie['name']} release year correct: {result['release_year']}")
                    else:
                        print(f"⚠ {movie['name']} year wrong: {result['release_year']} (expected {movie['expected_year']})")
                
                # Check director
                if 'director' in result:
                    if movie['expected_director'] in result['director'].lower():
                        print(f"✓ {movie['name']} director correct: {result['director']}")
                    else:
                        print(f"⚠ {movie['name']} director unexpected: {result['director']}")
                
                # Show other data
                for key, value in result.items():
                    if key not in ['release_year', 'director']:
                        print(f"  {key}: {value}")
                        
            else:
                print(f"✗ {movie['name']} failed with status {response.status_code}")
                
        except Exception as e:
            print(f"✗ {movie['name']} error: {e}")

def display_test_info():
    """Display information about the Wikipedia scraping test"""
    print("=== Titanic Wikipedia Scraping Test ===")
    print("This test will:")
    print("1. Send questions about Titanic (1997) to the API")
    print("2. API will scrape the Wikipedia page for Titanic")
    print("3. Extract information from the movie's infobox")
    print("4. Return structured data about the movie")
    print()
    print("Expected Titanic data:")
    print("  - Budget: ~$200 million")
    print("  - Worldwide Gross: >$2.2 billion")
    print("  - Director: James Cameron")
    print("  - Runtime: ~195 minutes")
    print("  - Release Year: 1997")
    print("  - Production Companies: Paramount, 20th Century Fox")

if __name__ == "__main__":
    display_test_info()
    print("\n" + "="*60 + "\n")
    
    # Test Titanic first
    test_titanic_wikipedia()
    
    # Test other movies
    test_other_movies()