import requests
import json

# Create test files
with open('test_edges.csv', 'w') as f:
    f.write("""source,target
Alice,Bob
Alice,Carol
Bob,Carol
Bob,David
Carol,David
Carol,Eve
David,Eve""")

with open('test_questions.txt', 'w') as f:
    f.write("""Return a JSON object with keys:
- edge_count: number
- highest_degree_node: string  
- average_degree: number
- density: number
- shortest_path_alice_eve: number
- network_graph: base64 PNG string under 100kB
- degree_histogram: base64 PNG string under 100kB

How many edges are in the network?
Which node has the highest degree?
What is the average degree of the network?
What is the network density?
What is the length of the shortest path between Alice and Eve?
Draw the network with nodes labelled and edges shown.
Plot the degree distribution as a bar chart with green bars.""")

# Send request
with open('test_edges.csv', 'rb') as ef, open('test_questions.txt', 'rb') as qf:
    files = [
        ('files', ('edges.csv', ef, 'text/csv')),
        ('files', ('questions.txt', qf, 'text/plain'))
    ]
    response = requests.post('http://localhost:8000/api/', files=files)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    try:
        result = response.json()
        print("Valid JSON response")
        print(f"Keys: {list(result.keys())}")
        if 'edge_count' in result:
            print(f"Edge count: {result['edge_count']}")
        if 'highest_degree_node' in result:
            print(f"Highest degree: {result['highest_degree_node']}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response: {response.text[:200]}")
else:
    print(f"Error: {response.text[:200]}")
