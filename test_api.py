#\!/usr/bin/env python3
"""Test script to verify API fixes"""

import requests
import json
import tempfile
import os

def test_network_analysis():
    """Test network analysis endpoint"""
    print("Testing network analysis...")
    
    # Create test data
    edges_csv = """source,target
Alice,Bob
Alice,Carol
Bob,Carol
Bob,David
Carol,David
Carol,Eve
David,Eve"""
    
    questions_txt = """Use the undirected network in edges.csv.

Return a JSON object with keys:
- edge_count: number
- highest_degree_node: string
- average_degree: number
- density: number
- shortest_path_alice_eve: number
- network_graph: base64 PNG string under 100kB
- degree_histogram: base64 PNG string under 100kB

Answer:
1. How many edges are in the network?
2. Which node has the highest degree?
3. What is the average degree of the network?
4. What is the network density?
5. What is the length of the shortest path between Alice and Eve?
6. Draw the network with nodes labelled and edges shown. Encode as base64 PNG.
7. Plot the degree distribution as a bar chart with green bars. Encode as base64 PNG."""
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(edges_csv)
        edges_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_txt)
        questions_file = f.name
    
    try:
        # Send request
        with open(edges_file, 'rb') as ef, open(questions_file, 'rb') as qf:
            files = [
                ('files', ('edges.csv', ef, 'text/csv')),
                ('files', ('questions.txt', qf, 'text/plain'))
            ]
            
            response = requests.post('http://localhost:8000/api/', files=files)
            
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("Response is valid JSON\!")
                print(f"Keys in response: {list(result.keys())}")
                
                # Check for expected keys
                expected = ['edge_count', 'highest_degree_node', 'average_degree', 
                           'density', 'shortest_path_alice_eve', 'network_graph', 'degree_histogram']
                missing = [k for k in expected if k not in result]
                if missing:
                    print(f"Missing keys: {missing}")
                else:
                    print("All expected keys present\!")
                    
                # Check values
                if 'edge_count' in result:
                    print(f"Edge count: {result['edge_count']} (expected: 7)")
                if 'highest_degree_node' in result:
                    print(f"Highest degree node: {result['highest_degree_node']}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response text: {response.text[:500]}")
        else:
            print(f"Error response: {response.text[:500]}")
            
    finally:
        # Clean up
        os.unlink(edges_file)
        os.unlink(questions_file)
    
    print("-" * 50)
    return response.status_code == 200

if __name__ == "__main__":
    test_network_analysis()
EOF < /dev/null
