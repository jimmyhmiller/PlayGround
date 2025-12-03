import re

# Parse the TypeScript SVG
with open('mega-complex-func5-pass0-ts.svg', 'r') as f:
    content = f.read()

# Extract all g transforms
positions = []
for match in re.finditer(r'<g transform="translate\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)">(.*?)</g>', content, re.DOTALL):
    x, y = float(match.group(1)), float(match.group(2))
    inner = match.group(3)
    
    # Try to extract block ID
    block_id = None
    id_match = re.search(r'Block (\d+)', inner)
    if id_match:
        block_id = int(id_match.group(1))
    
    positions.append({'x': x, 'y': y, 'block_id': block_id})

# Print positions grouped by Y coordinate (layer)
by_y = {}
for pos in positions:
    y = pos['y']
    if y not in by_y:
        by_y[y] = []
    by_y[y].append(pos)

for y in sorted(by_y.keys()):
    nodes = by_y[y]
    print(f"Y={y}: ", end="")
    for node in nodes:
        if node['block_id']:
            print(f"Block{node['block_id']}@x={node['x']}", end=" ")
        else:
            print(f"?@x={node['x']}", end=" ")
    print()
