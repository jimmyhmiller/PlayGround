import json

with open('mega-complex.json', 'r') as f:
    data = json.load(f)

# Find function 5, pass 0
func = data['functions'][5]
pass_data = func['passes'][0]

print(f"Function: {func['name']}")
print(f"Pass: {pass_data['name']}")
print()

if 'mir' in pass_data:
    blocks = pass_data['mir']['blocks']
    print(f"Total blocks: {len(blocks)}")
    print()
    
    # Check blocks 1, 6, 9, 12, 13, 14, 15
    problem_blocks = [1, 6, 9, 12, 13, 14, 15]
    
    for idx in problem_blocks:
        if idx < len(blocks):
            block = blocks[idx]
            print(f"Block {idx} (ID {block['id']}):")
            print(f"  Attributes: {block.get('attributes', [])}")
            print(f"  Successors: {[s for s in block.get('successors', [])]}")
            print(f"  Loop depth: {block.get('loopDepth', 0)}")
            print()
