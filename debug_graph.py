import io

csv_data = """source,target
Alice,Bob
Alice,Carol
Bob,Carol
Bob,David
Carol,David
Carol,Eve
David,Eve"""

# Build adjacency list manually
edges = []
for line in csv_data.strip().split('\n')[1:]:
    source, target = line.split(',')
    edges.append((source, target))

print("Edges:", edges)

# Count degrees
degrees = {}
for source, target in edges:
    degrees[source] = degrees.get(source, 0) + 1
    degrees[target] = degrees.get(target, 0) + 1

print("\nDegrees:")
for node in sorted(degrees.keys()):
    print(f"{node}: {degrees[node]}")

max_degree = max(degrees.values())
max_nodes = [node for node, deg in degrees.items() if deg == max_degree]
print(f"\nMax degree: {max_degree}")
print(f"Nodes with max degree: {max_nodes}")
