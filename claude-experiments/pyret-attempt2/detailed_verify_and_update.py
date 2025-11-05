#!/usr/bin/env python3
import subprocess
import re
import sys

BASE_DIR = "/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

def get_mismatch_reason(filename):
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
        if "✅ ASTs are identical" in output:
            return "✅ MATCHES official parser"
        
        # Check for specific difference patterns
        if "s-underscore" in output and "s-name" in output:
            return "❌ underscore handling (s-underscore vs s-name)"
        
        # Look for other patterns in diff
        if '"type":' in output:
            # Extract first type difference
            match = re.search(r'<\s+"type":\s+"([^"]+)".*?>\s+"type":\s+"([^"]+)"', output, re.DOTALL)
            if match:
                return f"❌ AST difference ({match.group(1)} vs {match.group(2)})"
        
        return "❌ PARSES but doesn't match official"
        
    except subprocess.TimeoutExpired:
        return "❌ comparison timeout"
    except Exception as e:
        return f"❌ comparison error: {str(e)[:40]}"

# Read current file
with open('bulk_test_results/failing_files.txt', 'r') as f:
    lines = f.readlines()

output_lines = []
count = 0
verified = 0
underscore_issues = 0
other_mismatches = 0

for line in lines:
    line = line.rstrip('\n')
    
    # Only process SUCCESS (verified) lines
    if "SUCCESS (verified)" not in line:
        output_lines.append(line)
        continue
    
    filename = line.split('#')[0].strip()
    count += 1
    
    if count % 5 == 0:
        print(f"Checking {count}...", file=sys.stderr)
    
    reason = get_mismatch_reason(filename)
    output_lines.append(f"{filename}  # {reason}")
    
    if "MATCHES" in reason:
        verified += 1
        print(f"✅ {filename}", file=sys.stderr)
    elif "underscore" in reason:
        underscore_issues += 1
        print(f"🔸 {filename} - underscore issue", file=sys.stderr)
    else:
        other_mismatches += 1
        print(f"❌ {filename} - {reason}", file=sys.stderr)

# Write updated file
with open('bulk_test_results/failing_files.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

print("\n" + "=" * 70, file=sys.stderr)
print(f"Summary:", file=sys.stderr)
print(f"  ✅ Perfect matches: {verified}", file=sys.stderr)
print(f"  🔸 Underscore issues: {underscore_issues}", file=sys.stderr)
print(f"  ❌ Other mismatches: {other_mismatches}", file=sys.stderr)
print(f"  Total tested: {count}", file=sys.stderr)
print("=" * 70, file=sys.stderr)
