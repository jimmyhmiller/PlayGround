#!/usr/bin/env python3
import re
import os
from collections import Counter

BASE_DIR = "/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

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

def check_file_for_type_alias(filename, line_num):
    """Check if the error line contains 'type Name = '"""
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        return False
    
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
            if 0 <= line_num-1 < len(lines):
                line = lines[line_num-1].strip()
                # Check for type alias pattern
                if re.match(r'^type\s+\w+\s*=', line) or re.match(r'^type\s+\w+\s*<', line):
                    return True
    except:
        pass
    return False

def check_file_for_newtype(filename, line_num):
    """Check if the error line contains 'newtype Name: '"""
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        return False
    
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
            if 0 <= line_num-1 < len(lines):
                line = lines[line_num-1].strip()
                if re.match(r'^newtype\s+\w+:', line):
                    return True
    except:
        pass
    return False

def categorize_error(filename, error):
    """Determine which missing feature causes the error"""
    
    if error.startswith('SUCCESS'):
        return 'SUCCESS - Parser now handles this!'
    
    # Specific error patterns
    error_lower = error.lower()
    
    if 'colon, found: from' in error_lower:
        return 'MISSING: import...from syntax'
    
    if 'colon, found: lbrace' in error_lower:
        return 'MISSING: provide { } block syntax'
    
    if 'name, found: type' in error_lower:
        return 'MISSING: type in provide'
    
    if 'name, found: times' in error_lower:
        return 'MISSING: * in provide list'
    
    if 'name, found: module' in error_lower:
        return 'MISSING: module in provide'
    
    if 'name, found: data' in error_lower:
        return 'MISSING: data in provide'
        
    if "expected 'as' after import module" in error_lower:
        return 'MISSING: import without as'
    
    if 'end, found: as' in error_lower:
        return 'MISSING: provide...as syntax'
    
    if 'end, found: dot' in error_lower:
        return 'MISSING: provide from syntax'
    
    # "Unexpected tokens after program end" - check the actual file
    if 'unexpected tokens after program end' in error_lower:
        match = re.search(r'location:\s*"(\d+):(\d+):(\d+)"', error)
        if match:
            line_num = int(match.group(1))
            
            # Check for type alias
            if check_file_for_type_alias(filename, line_num):
                return 'MISSING: type alias (type Name = Type)'
            
            # Check for newtype
            if check_file_for_newtype(filename, line_num):
                return 'MISSING: newtype declarations'
            
            # Location-based categorization
            if line_num <= 3:
                return 'MISSING: Prelude feature (#lang, provide-types, etc.)'
            elif line_num <= 15:
                return 'MISSING: Import/provide/type feature'
            else:
                return 'MISSING: Runtime feature (needs manual check)'
        return 'MISSING: Unknown parse error'
    
    return f'UNKNOWN ERROR'

# Process all files
output_lines = []
for filename, error in files:
    category = categorize_error(filename, error)
    output_lines.append(f"{filename}  # {category}")

# Write annotated output
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
print(f"\nTotal files analyzed: {len(files)}")
print(f"✅ Parsing successfully: {summary.get('SUCCESS - Parser now handles this!', 0)}")
print(f"📝 Type alias needed: {summary.get('MISSING: type alias (type Name = Type)', 0)}")
print(f"🔧 Advanced import/export: {sum(v for k,v in summary.items() if 'import' in k.lower() or 'provide' in k.lower())}")
print(f"❓ Need manual check: {sum(v for k,v in summary.items() if 'manual' in k.lower() or 'Runtime' in k)}")
