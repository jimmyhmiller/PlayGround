#!/usr/bin/env python3
import re
from collections import Counter

# Read failure_analysis.txt
with open('bulk_test_results/failure_analysis.txt', 'r') as f:
    content = f.read()

# Parse into file entries
entries = re.split(r'^=== (.+?) ===$', content, flags=re.MULTILINE)[1:]
files = []

for i in range(0, len(entries), 2):
    if i+1 < len(entries):
        filename = entries[i]
        error = entries[i+1].strip()
        files.append((filename, error))

def categorize_error(filename, error):
    """Determine which missing feature causes the error"""
    
    if error.startswith('SUCCESS'):
        return 'SUCCESS - Parser now handles this!'
    
    # Specific error patterns (case-insensitive contains)
    error_lower = error.lower()
    
    if 'colon, found: from' in error_lower:
        return 'MISSING: import...from syntax (import x from file("..."))'
    
    if 'colon, found: lbrace' in error_lower:
        return 'MISSING: provide { } block syntax'
    
    if 'name, found: type' in error_lower:
        return 'MISSING: type keyword in provide (provide: type T end)'
    
    if 'name, found: times' in error_lower:
        return 'MISSING: * in provide list (provide: *, type T end)'
    
    if 'name, found: module' in error_lower:
        return 'MISSING: module keyword in provide'
    
    if 'name, found: data' in error_lower:
        return 'MISSING: data keyword in provide (provide: data D end)'
        
    if "expected 'as' after import module" in error_lower:
        return 'MISSING: import without as (import file("x"))'
    
    if 'end, found: as' in error_lower:
        return 'MISSING: provide...as syntax (provide x as y)'
    
    if 'end, found: dot' in error_lower:
        return 'MISSING: Unknown syntax with dot'
    
    # Generic "Unexpected tokens after program end" - need to check file
    if 'unexpected tokens after program end' in error_lower:
        # Extract location
        match = re.search(r'location:\s*"(\d+):(\d+):(\d+)"', error)
        if match:
            line_num = int(match.group(1))
            if line_num <= 3:
                return 'MISSING: Advanced prelude syntax (lines 1-3)'
            elif line_num <= 10:
                return 'MISSING: Advanced import/provide (lines 4-10)'
            else:
                return 'MISSING: Unknown feature (check file manually)'
        return 'MISSING: Unknown feature'
    
    return f'UNKNOWN ERROR: {error[:80]}'

# Create annotated output
output_lines = []
for filename, error in files:
    category = categorize_error(filename, error)
    output_lines.append(f"{filename}  # {category}")

# Write to failing_files.txt with annotations
with open('bulk_test_results/failing_files_annotated.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

# Print summary
categories = [categorize_error(f, e) for f, e in files]
summary = Counter(categories)

print("=" * 80)
print("FAILURE CATEGORIZATION SUMMARY")
print("=" * 80)
for category, count in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"{count:3d}  {category}")
print("=" * 80)
print(f"Total files: {len(files)}")
print(f"Parsing successfully: {summary['SUCCESS - Parser now handles this!']}")
print(f"Need investigation: {sum(v for k,v in summary.items() if 'Unknown' in k or 'check file' in k)}")
