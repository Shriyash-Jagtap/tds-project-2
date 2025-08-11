#!/usr/bin/env python3
"""
Test script for network analysis API
Tests the API with network edge data to analyze graph properties
"""
import requests
import json
import base64
from io import BytesIO

def test_network_api():
    """Test the API with network edge data"""
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    try:
        # Prepare files to upload
        files = []
        
        # Add questions.txt
        with open('network-questions.txt', 'r') as f:
            questions_content = f.read()
        files.append(('files', ('questions.txt', questions_content, 'text/plain')))
        
        # Add edges CSV
        with open('edges.csv', 'rb') as f:
            csv_content = f.read()
        files.append(('files', ('edges.csv', csv_content, 'text/csv')))
        
        print("Sending network analysis request to API...")
        response = requests.post(url, files=files, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Network Analysis Results ===")
            
            # Expected fields for network analysis
            expected_fields = [
                'edge_count',
                'highest_degree_node', 
                'average_degree',
                'density',
                'shortest_path_alice_eve',
                'network_graph',
                'degree_histogram'
            ]
            
            for field in expected_fields:
                if field in result:
                    value = result[field]
                    if isinstance(value, str) and value.startswith('data:image'):
                        print(f"✓ {field}: [base64 image - {len(value)} chars]")
                    else:
                        print(f"✓ {field}: {value}")
                else:
                    print(f"✗ {field}: MISSING")
            
            # Validate the results
            print("\n=== Validation ===")
            
            # Check edge count (should be 7 based on the CSV)
            if 'edge_count' in result:
                expected_edges = 7
                if result['edge_count'] == expected_edges:
                    print(f"✓ Edge count correct: {expected_edges}")
                else:
                    print(f"✗ Edge count mismatch: got {result['edge_count']}, expected {expected_edges}")
            
            # Check highest degree node (Bob has 4 connections)
            if 'highest_degree_node' in result:
                expected_node = 'Bob'
                if result['highest_degree_node'] == expected_node:
                    print(f"✓ Highest degree node correct: {expected_node}")
                else:
                    print(f"✗ Highest degree node mismatch: got {result['highest_degree_node']}, expected {expected_node}")
            
            # Check average degree (should be 2.8 for this network)
            if 'average_degree' in result:
                expected_avg = 2.8
                if abs(result['average_degree'] - expected_avg) < 0.01:
                    print(f"✓ Average degree correct: {expected_avg}")
                else:
                    print(f"✗ Average degree mismatch: got {result['average_degree']}, expected {expected_avg}")
            
            # Check density (7 edges / (5*4/2) = 0.7)
            if 'density' in result:
                expected_density = 0.7
                if abs(result['density'] - expected_density) < 0.01:
                    print(f"✓ Network density correct: {expected_density}")
                else:
                    print(f"✗ Network density mismatch: got {result['density']}, expected {expected_density}")
            
            # Check shortest path (Alice -> Eve should be 2)
            if 'shortest_path_alice_eve' in result:
                expected_path = 2
                if result['shortest_path_alice_eve'] == expected_path:
                    print(f"✓ Shortest path Alice->Eve correct: {expected_path}")
                else:
                    print(f"✗ Shortest path mismatch: got {result['shortest_path_alice_eve']}, expected {expected_path}")
            
            # Check if visualizations are present
            if 'network_graph' in result and result['network_graph']:
                print("✓ Network graph visualization generated")
            else:
                print("✗ Network graph visualization missing")
                
            if 'degree_histogram' in result and result['degree_histogram']:
                print("✓ Degree histogram visualization generated")
            else:
                print("✗ Degree histogram visualization missing")
                
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:8000")
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}")
        print("Make sure edges.csv and network-questions.txt are in the current directory")
    except Exception as e:
        print(f"Error: {e}")

def display_network_info():
    """Display information about the test network"""
    print("\n=== Test Network Information ===")
    print("Network edges from edges.csv:")
    print("  Alice -> Bob")
    print("  Alice -> Carol")
    print("  Bob -> Carol")
    print("  Bob -> David")
    print("  Bob -> Eve")
    print("  Carol -> David")
    print("  David -> Eve")
    print("\nNode degrees:")
    print("  Alice: 2 (Bob, Carol)")
    print("  Bob: 4 (Alice, Carol, David, Eve)")
    print("  Carol: 3 (Alice, Bob, David)")
    print("  David: 3 (Bob, Carol, Eve)")
    print("  Eve: 2 (Bob, David)")

if __name__ == "__main__":
    display_network_info()
    print("\n" + "="*50 + "\n")
    test_network_api()