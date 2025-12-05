#!/usr/bin/env python3
"""
Generate a very large universal format graph for testing.

This creates a complex control flow graph with:
- Multiple nested loops
- Many blocks (200+)
- Complex control flow patterns
- Backedges at multiple levels
"""

import json

def generate_large_universal():
    blocks = []
    block_id = 0

    # Entry block
    blocks.append({
        "id": str(block_id),
        "attributes": ["entry"],
        "loopDepth": 0,
        "predecessors": [],
        "successors": [str(block_id + 1)],
        "instructions": [
            {"opcode": "Parameter", "attributes": ["arg0"], "type": "int32"},
            {"opcode": "Parameter", "attributes": ["arg1"], "type": "int32"},
            {"opcode": "Parameter", "attributes": ["arg2"], "type": "int32"},
            {"opcode": "Constant", "attributes": [], "type": "int32"},
            {"opcode": "Constant", "attributes": [], "type": "int32"},
        ]
    })
    block_id += 1

    # Outer loop header (depth 1)
    outer_loop_header = block_id
    blocks.append({
        "id": str(block_id),
        "attributes": ["loopheader"],
        "loopDepth": 1,
        "predecessors": [str(block_id - 1), str(block_id + 40)],  # from entry and backedge
        "successors": [str(block_id + 1), str(block_id + 41)],  # to body or exit
        "instructions": [
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Compare", "attributes": ["LessThan"], "type": "bool"},
            {"opcode": "Branch", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # First inner loop (depth 2) - 15 blocks
    first_inner_header = block_id
    blocks.append({
        "id": str(block_id),
        "attributes": ["loopheader"],
        "loopDepth": 2,
        "predecessors": [str(outer_loop_header), str(block_id + 14)],
        "successors": [str(block_id + 1), str(block_id + 15)],
        "instructions": [
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Compare", "attributes": ["LessThan"], "type": "bool"},
            {"opcode": "Branch", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Inner loop body - chain of blocks
    for i in range(13):
        is_last = (i == 12)
        blocks.append({
            "id": str(block_id),
            "attributes": [],
            "loopDepth": 2,
            "predecessors": [str(block_id - 1)],
            "successors": [str(block_id + 1)],
            "instructions": [
                {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
                {"opcode": "Mul", "attributes": ["Commutative"], "type": "int32"},
                {"opcode": "Sub", "attributes": [], "type": "int32"},
                {"opcode": "Store", "attributes": [], "type": "void"},
                {"opcode": "Load", "attributes": [], "type": "int32"},
            ]
        })
        block_id += 1

    # Backedge block for first inner loop
    blocks.append({
        "id": str(block_id),
        "attributes": ["backedge"],
        "loopDepth": 2,
        "predecessors": [str(block_id - 1)],
        "successors": [str(first_inner_header)],
        "instructions": [
            {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
            {"opcode": "Goto", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Second inner loop (depth 2) - 20 blocks with nested loop (depth 3)
    second_inner_header = block_id
    blocks.append({
        "id": str(block_id),
        "attributes": ["loopheader"],
        "loopDepth": 2,
        "predecessors": [str(first_inner_header + 15), str(block_id + 19)],
        "successors": [str(block_id + 1), str(block_id + 20)],
        "instructions": [
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Compare", "attributes": ["LessThan"], "type": "bool"},
            {"opcode": "Branch", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Some blocks before nested loop (depth 3)
    for i in range(3):
        blocks.append({
            "id": str(block_id),
            "attributes": [],
            "loopDepth": 2,
            "predecessors": [str(block_id - 1)],
            "successors": [str(block_id + 1)],
            "instructions": [
                {"opcode": "Load", "attributes": [], "type": "int64"},
                {"opcode": "Add", "attributes": ["Commutative"], "type": "int64"},
                {"opcode": "Store", "attributes": [], "type": "void"},
            ]
        })
        block_id += 1

    # Nested loop header (depth 3)
    nested_loop_header = block_id
    blocks.append({
        "id": str(block_id),
        "attributes": ["loopheader"],
        "loopDepth": 3,
        "predecessors": [str(block_id - 1), str(block_id + 8)],
        "successors": [str(block_id + 1), str(block_id + 9)],
        "instructions": [
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Compare", "attributes": ["LessThan"], "type": "bool"},
            {"opcode": "Branch", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Nested loop body (depth 3) - 7 blocks
    for i in range(7):
        blocks.append({
            "id": str(block_id),
            "attributes": [],
            "loopDepth": 3,
            "predecessors": [str(block_id - 1)],
            "successors": [str(block_id + 1)],
            "instructions": [
                {"opcode": "Mul", "attributes": ["Commutative"], "type": "int32"},
                {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
                {"opcode": "Div", "attributes": [], "type": "int32"},
            ]
        })
        block_id += 1

    # Backedge for nested loop (depth 3)
    blocks.append({
        "id": str(block_id),
        "attributes": ["backedge"],
        "loopDepth": 3,
        "predecessors": [str(block_id - 1)],
        "successors": [str(nested_loop_header)],
        "instructions": [
            {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
            {"opcode": "Goto", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Exit from nested loop - more blocks in second inner loop (depth 2)
    for i in range(6):
        blocks.append({
            "id": str(block_id),
            "attributes": [],
            "loopDepth": 2,
            "predecessors": [str(block_id - 1)],
            "successors": [str(block_id + 1)],
            "instructions": [
                {"opcode": "Load", "attributes": [], "type": "int32"},
                {"opcode": "Store", "attributes": [], "type": "void"},
                {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
            ]
        })
        block_id += 1

    # Backedge for second inner loop
    blocks.append({
        "id": str(block_id),
        "attributes": ["backedge"],
        "loopDepth": 2,
        "predecessors": [str(block_id - 1)],
        "successors": [str(second_inner_header)],
        "instructions": [
            {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
            {"opcode": "Goto", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Third inner loop (depth 2) - simpler, 10 blocks
    third_inner_header = block_id
    blocks.append({
        "id": str(block_id),
        "attributes": ["loopheader"],
        "loopDepth": 2,
        "predecessors": [str(second_inner_header + 20), str(block_id + 9)],
        "successors": [str(block_id + 1), str(block_id + 10)],
        "instructions": [
            {"opcode": "Phi", "attributes": [], "type": "float64"},
            {"opcode": "Compare", "attributes": ["LessThan"], "type": "bool"},
            {"opcode": "Branch", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Third loop body
    for i in range(8):
        blocks.append({
            "id": str(block_id),
            "attributes": [],
            "loopDepth": 2,
            "predecessors": [str(block_id - 1)],
            "successors": [str(block_id + 1)],
            "instructions": [
                {"opcode": "FAdd", "attributes": [], "type": "float64"},
                {"opcode": "FMul", "attributes": [], "type": "float64"},
                {"opcode": "FDiv", "attributes": [], "type": "float64"},
            ]
        })
        block_id += 1

    # Backedge for third inner loop
    blocks.append({
        "id": str(block_id),
        "attributes": ["backedge"],
        "loopDepth": 2,
        "predecessors": [str(block_id - 1)],
        "successors": [str(third_inner_header)],
        "instructions": [
            {"opcode": "FAdd", "attributes": [], "type": "float64"},
            {"opcode": "Goto", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Backedge for outer loop
    blocks.append({
        "id": str(block_id),
        "attributes": ["backedge"],
        "loopDepth": 1,
        "predecessors": [str(third_inner_header + 10)],
        "successors": [str(outer_loop_header)],
        "instructions": [
            {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
            {"opcode": "Goto", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Exit block
    blocks.append({
        "id": str(block_id),
        "attributes": ["exit"],
        "loopDepth": 0,
        "predecessors": [str(outer_loop_header)],
        "successors": [],
        "instructions": [
            {"opcode": "Return", "attributes": [], "type": "int32"},
        ]
    })
    block_id += 1

    # Add more complexity: a separate function with many sequential blocks
    # This creates a very long edge chain that will require many dummy nodes

    entry2 = block_id
    blocks.append({
        "id": str(block_id),
        "attributes": [],
        "loopDepth": 0,
        "predecessors": [],
        "successors": [str(block_id + 50)],  # Jump far ahead
        "instructions": [
            {"opcode": "Call", "attributes": [], "type": "void"},
            {"opcode": "Constant", "attributes": [], "type": "int32"},
        ]
    })
    block_id += 1

    # Fill in the gap with intermediate blocks
    for i in range(49):
        blocks.append({
            "id": str(block_id),
            "attributes": [],
            "loopDepth": 0,
            "predecessors": [str(block_id - 1)] if i > 0 else [],
            "successors": [str(block_id + 1)],
            "instructions": [
                {"opcode": "Nop", "attributes": [], "type": "void"},
                {"opcode": "Add", "attributes": ["Commutative"], "type": "int32"},
            ]
        })
        block_id += 1

    # Target of the long jump
    blocks.append({
        "id": str(block_id),
        "attributes": [],
        "loopDepth": 0,
        "predecessors": [str(entry2), str(block_id - 1)],
        "successors": [str(block_id + 1)],
        "instructions": [
            {"opcode": "Phi", "attributes": [], "type": "int32"},
            {"opcode": "Store", "attributes": [], "type": "void"},
        ]
    })
    block_id += 1

    # Add more sequential blocks with occasional branches
    for i in range(100):
        if i % 20 == 19:
            # Branch block
            blocks.append({
                "id": str(block_id),
                "attributes": [],
                "loopDepth": 0,
                "predecessors": [str(block_id - 1)],
                "successors": [str(block_id + 1), str(block_id + 5)],
                "instructions": [
                    {"opcode": "Compare", "attributes": ["Equal"], "type": "bool"},
                    {"opcode": "Branch", "attributes": [], "type": "void"},
                ]
            })
            block_id += 1

            # Then path (3 blocks)
            for j in range(3):
                blocks.append({
                    "id": str(block_id),
                    "attributes": [],
                    "loopDepth": 0,
                    "predecessors": [str(block_id - 1)],
                    "successors": [str(block_id + 1)],
                    "instructions": [
                        {"opcode": "Add", "attributes": [], "type": "int32"},
                        {"opcode": "Mul", "attributes": [], "type": "int32"},
                    ]
                })
                block_id += 1

            # Merge block
            blocks.append({
                "id": str(block_id),
                "attributes": [],
                "loopDepth": 0,
                "predecessors": [str(block_id - 1), str(block_id - 4)],
                "successors": [str(block_id + 1)],
                "instructions": [
                    {"opcode": "Phi", "attributes": [], "type": "int32"},
                ]
            })
            block_id += 1
        else:
            # Regular sequential block
            blocks.append({
                "id": str(block_id),
                "attributes": [],
                "loopDepth": 0,
                "predecessors": [str(block_id - 1)],
                "successors": [str(block_id + 1)],
                "instructions": [
                    {"opcode": "Load", "attributes": [], "type": "int64"},
                    {"opcode": "Add", "attributes": ["Commutative"], "type": "int64"},
                    {"opcode": "Store", "attributes": [], "type": "void"},
                    {"opcode": "Call", "attributes": [], "type": "void"},
                ]
            })
            block_id += 1

    # Final return block
    blocks.append({
        "id": str(block_id),
        "attributes": [],
        "loopDepth": 0,
        "predecessors": [str(block_id - 1)],
        "successors": [],
        "instructions": [
            {"opcode": "Return", "attributes": [], "type": "void"},
        ]
    })

    return {
        "format": "codegraph-v1",
        "compiler": "synthetic",
        "metadata": {
            "name": "veryLargeComplexFunction",
            "optimization_level": 3,
            "blocks": len(blocks),
            "description": "Large synthetic test case with nested loops, long edges, and complex control flow"
        },
        "blocks": blocks
    }

if __name__ == "__main__":
    graph = generate_large_universal()
    with open("examples/large-universal.json", "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Generated large universal graph with {len(graph['blocks'])} blocks")
    print(f"Written to examples/large-universal.json")

    # Print some statistics
    max_loop_depth = max(b['loopDepth'] for b in graph['blocks'])
    loop_headers = sum(1 for b in graph['blocks'] if 'loopheader' in b['attributes'])
    backedges = sum(1 for b in graph['blocks'] if 'backedge' in b['attributes'])

    print(f"\nStatistics:")
    print(f"  Max loop depth: {max_loop_depth}")
    print(f"  Loop headers: {loop_headers}")
    print(f"  Backedge blocks: {backedges}")
