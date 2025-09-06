#!/usr/bin/env python3
"""
TypeScript to Rust Snapshot Porting Tool

This tool takes TypeScript snapshot outputs and creates corresponding Rust snapshot 
files so we can run tests and see the exact diffs between our implementation and 
the original.
"""

import os
import json
import re
from pathlib import Path

# Paths
RUST_DIR = Path("/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust")
TYPESCRIPT_DIR = Path("/Users/jimmyhmiller/Documents/Code/open-source/iongraph")

RUST_SNAPSHOTS_DIR = RUST_DIR / "src" / "snapshots"
TS_SNAPSHOTS_DIR = TYPESCRIPT_DIR / "src" / "test" / "__snapshots__"

def parse_typescript_snapshots():
    """Parse TypeScript snapshot file and extract all test data."""
    ts_snap_file = TS_SNAPSHOTS_DIR / "Graph.snapshots.test.ts.snap"
    if not ts_snap_file.exists():
        print(f"❌ TypeScript snapshots not found at: {ts_snap_file}")
        return {}
    
    with open(ts_snap_file, 'r') as f:
        content = f.read()
    
    snapshots = {}
    
    # Parse exports pattern: exports[`Graph Snapshots > Category > Test Name 1`] = `content`;
    pattern = r"exports\[`Graph Snapshots > (.*?) > (.*?) 1`\] = `(.*?)`;(?=\n\nexports|\n\n$|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"📖 Found {len(matches)} TypeScript snapshot tests")
    
    for category, test_name, test_content in matches:
        key = f"{category} > {test_name}"
        
        # Try to parse as JSON
        try:
            parsed_content = json.loads(test_content.strip())
        except json.JSONDecodeError:
            # Keep as raw string if not JSON
            parsed_content = test_content.strip()
        
        snapshots[key] = {
            'category': category,
            'test_name': test_name,
            'content': parsed_content,
            'raw_content': test_content.strip()
        }
        
        print(f"  ✓ {key}")
    
    return snapshots

def json_to_yaml(obj, indent=0):
    """Convert JSON object to YAML format (simple implementation)."""
    spaces = "  " * indent
    
    if obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        lines = []
        for item in obj:
            item_yaml = json_to_yaml(item, indent + 1)
            if isinstance(item, dict):
                lines.append(f"{spaces}- " + item_yaml.lstrip())
            else:
                lines.append(f"{spaces}- {item_yaml}")
        return "\n".join(lines)
    elif isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for key, value in obj.items():
            value_yaml = json_to_yaml(value, indent + 1)
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{spaces}{key}:")
                if isinstance(value, dict):
                    for line in value_yaml.split('\n'):
                        if line.strip():
                            lines.append(f"  {line}")
                else:
                    lines.append(f"  {value_yaml}")
            else:
                lines.append(f"{spaces}{key}: {value_yaml}")
        return "\n".join(lines)
    else:
        return str(obj)

def create_rust_snapshot_file(test_name, content, test_function_name):
    """Create a Rust snapshot file with the TypeScript content as the expected result."""
    
    # Create the snapshot file content
    if isinstance(content, str):
        # Raw string content
        snapshot_content = f"""---
source: src/graph.rs
expression: {test_function_name.replace('test_', '').replace('_snapshot', '')}
---
{content}
"""
    else:
        # JSON content - convert to YAML
        yaml_content = json_to_yaml(content)
        snapshot_content = f"""---
source: src/graph.rs
expression: {test_function_name.replace('test_', '').replace('_snapshot', '')}
---
{yaml_content}
"""
    
    # Write to snapshot file
    snap_filename = f"iongraph_rust__graph__tests__{test_function_name}.snap"
    snap_path = RUST_SNAPSHOTS_DIR / snap_filename
    
    with open(snap_path, 'w') as f:
        f.write(snapshot_content)
    
    print(f"  ✓ Created: {snap_filename}")
    return snap_path

def map_typescript_to_rust_tests(ts_snapshots):
    """Map TypeScript snapshot tests to Rust test function names."""
    mappings = {
        # Block Intermediate Representation Snapshots
        "Block Intermediate Representation Snapshots > should create expected block IR for simple pass": "test_block_ir_snapshot_simple",
        "Block Intermediate Representation Snapshots > should create expected block IR for loop pass": "test_block_ir_snapshot_loop", 
        "Block Intermediate Representation Snapshots > should create expected block IR for complex control flow": "test_block_ir_snapshot_complex",
        
        # Layout Node Structure Snapshots  
        "Layout Node Structure Snapshots > should create expected layout structure for simple pass": "test_layout_algorithm_snapshot_simple",
        "Layout Node Structure Snapshots > should create expected layout structure for loop pass": "test_layout_algorithm_snapshot_loop",
        "Layout Node Structure Snapshots > should create expected layout structure for complex control flow": "test_layout_algorithm_snapshot_complex",
        
        # Pan and Zoom State Snapshots
        "Pan and Zoom State Snapshots > should create expected pan/zoom state": "test_pan_zoom_snapshot",
        
        # Navigation State Snapshots
        "Navigation State Snapshots > should create expected navigation state after selection": "test_navigation_state_snapshot_selection", 
        "Navigation State Snapshots > should create expected navigation state after navigation sequence": "test_navigation_state_snapshot_sequence",
        
        # Instruction Highlighting Snapshots
        "Instruction Highlighting Snapshots > should create expected highlighting state": "test_instruction_highlighting_snapshot",
    }
    
    return mappings

def main():
    print("🔄 TypeScript to Rust Snapshot Porting Tool")
    print("=" * 60)
    print()
    
    # Ensure snapshots directory exists
    RUST_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parse TypeScript snapshots
    print("📖 Parsing TypeScript snapshots...")
    ts_snapshots = parse_typescript_snapshots()
    
    if not ts_snapshots:
        print("❌ No TypeScript snapshots found!")
        return
    
    print()
    
    # Get mappings
    mappings = map_typescript_to_rust_tests(ts_snapshots)
    
    print("🔄 Creating Rust snapshot files from TypeScript expected output...")
    created_files = []
    
    for ts_key, rust_test_name in mappings.items():
        if ts_key in ts_snapshots:
            ts_data = ts_snapshots[ts_key]
            snap_path = create_rust_snapshot_file(
                rust_test_name, 
                ts_data['content'], 
                rust_test_name
            )
            created_files.append(snap_path)
        else:
            print(f"  ⚠️  TypeScript test not found: {ts_key}")
    
    print()
    print(f"✅ Successfully created {len(created_files)} Rust snapshot files")
    print()
    print("🎯 Next steps:")
    print("1. Run 'cargo test' to see the diffs between our Rust output and TypeScript expected")
    print("2. Fix the Rust implementation to match TypeScript output exactly")
    print("3. Use 'cargo insta review' to see detailed diffs")
    print()
    print("📁 Created snapshot files:")
    for path in created_files:
        print(f"  - {path.name}")

if __name__ == "__main__":
    main()