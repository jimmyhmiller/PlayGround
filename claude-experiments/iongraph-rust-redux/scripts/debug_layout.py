#!/usr/bin/env python3
"""
Compare layout node positions between TypeScript and Rust implementations
"""
import re
import json

def parse_svg_positions(filename):
    """Extract block positions from SVG file"""
    with open(filename, 'r') as f:
        content = f.read()
    
    positions = []
    # Find all transform="translate(x, y)" patterns
    for match in re.finditer(r'<g transform="translate\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)">', content):
        x, y = float(match.group(1)), float(match.group(2))
        positions.append({'x': x, 'y': y})
    
    return positions

def compare_positions(ts_file, rust_file):
    ts_pos = parse_svg_positions(ts_file)
    rust_pos = parse_svg_positions(rust_file)
    
    print(f"TypeScript: {len(ts_pos)} nodes")
    print(f"Rust: {len(rust_pos)} nodes")
    print()
    
    mismatches = []
    for i, (ts, rust) in enumerate(zip(ts_pos, rust_pos)):
        dx = abs(ts['x'] - rust['x'])
        dy = abs(ts['y'] - rust['y'])
        
        if dx > 0.1 or dy > 0.1:
            mismatches.append({
                'index': i,
                'ts': ts,
                'rust': rust,
                'dx': dx,
                'dy': dy
            })
            print(f"❌ Node {i}: TS ({ts['x']:.1f}, {ts['y']:.1f}) vs Rust ({rust['x']:.1f}, {rust['y']:.1f}) - diff: ({dx:.1f}, {dy:.1f})")
        else:
            print(f"✅ Node {i}: ({ts['x']:.1f}, {ts['y']:.1f})")
    
    print(f"\n{len(ts_pos) - len(mismatches)}/{len(ts_pos)} nodes match perfectly")
    return mismatches

if __name__ == '__main__':
    mismatches = compare_positions('mega-complex-func5-pass0-ts.svg', 'function5-pass0.svg')
    
    if mismatches:
        print("\n=== Mismatched Nodes ===")
        for m in mismatches:
            print(f"Node {m['index']}: {m['dx']:.1f}px X-axis, {m['dy']:.1f}px Y-axis difference")
