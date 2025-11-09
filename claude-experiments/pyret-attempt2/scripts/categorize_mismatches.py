#!/usr/bin/env python3
import subprocess
import re
from collections import Counter

BASE_DIR = "/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

def analyze_mismatch(filename):
    """Run comparison and determine what's different"""
    full_path = f"{BASE_DIR}/{filename}"
    
    try:
        result = subprocess.run(
            ["./compare_parsers.sh", full_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout + result.stderr
        
        # Check if identical
        if "✅ ASTs are identical" in output or "ASTs match" in output:
            return "✅ MATCHES"
        
        # Check for specific issues
        if 's-underscore' in output and '"name": "_"' in output:
            return "🔸 underscore (s-underscore vs s-name)"
        
        if 's-dot' in output or 's-get-bang' in output:
            return "🔸 dot/bang operator handling"
        
        if 's-block' in output:
            return "🔸 block expression structure"
        
        if '"l":' in output and '"loc":' in output:
            return "🔸 location field naming (l vs loc)"
        
        # Generic categorization
        return "❌ other AST difference"
        
    except subprocess.TimeoutExpired:
        return "⏱️ timeout"
    except Exception as e:
        return f"💥 error"

# Read file
with open('bulk_test_results/failing_files.txt', 'r') as f:
    lines = f.readlines()

print("Analyzing files that parse but don't match...\n")

results = []
count = 0

for line in lines:
    if "PARSES but doesn't match" in line:
        filename = line.split('#')[0].strip()
        count += 1
        print(f"Checking {count}: {filename}...", end=" ")
        category = analyze_mismatch(filename)
        results.append((filename, category))
        print(category)

# Update file with detailed categories
output_lines = []
result_dict = {f: c for f, c in results}

for line in lines:
    line = line.rstrip('\n')
    filename = line.split('#')[0].strip()
    
    if filename in result_dict:
        output_lines.append(f"{filename}  # {result_dict[filename]}")
    else:
        output_lines.append(line)

with open('bulk_test_results/failing_files.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

# Print summary
categories = [c for _, c in results]
summary = Counter(categories)

print("\n" + "=" * 70)
print("MISMATCH CATEGORIZATION:")
print("=" * 70)
for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"{count:3d}  {cat}")
print("=" * 70)
