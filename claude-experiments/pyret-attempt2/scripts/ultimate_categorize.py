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

def check_file_content(filename, line_num, patterns):
    """Check if the error line matches any of the given patterns"""
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        return None
    
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
            if 0 <= line_num-1 < len(lines):
                line = lines[line_num-1].strip()
                for pattern_name, pattern_regex in patterns.items():
                    if re.search(pattern_regex, line):
                        return pattern_name
    except:
        pass
    return None

def categorize_error(filename, error):
    """Determine which missing feature causes the error"""
    
    if error.startswith('SUCCESS'):
        return 'SUCCESS - Parses correctly now!'
    
    error_lower = error.lower()
    
    # Specific error patterns from tokenizer/parser
    if 'colon, found: from' in error_lower:
        return 'import from (import x, y from file(...))'
    
    if 'colon, found: lbrace' in error_lower:
        return 'provide block (provide { x: x, f: f } end)'
    
    if 'name, found: type' in error_lower:
        return 'type in provide (provide: type T, data D end)'
    
    if 'name, found: times' in error_lower:
        return 'asterisk in provide (provide: *, type T end)'
    
    if 'name, found: module' in error_lower:
        return 'module in provide (provide: module M end)'
    
    if 'name, found: data' in error_lower:
        return 'data in provide (provide: data D end)'
        
    if "expected 'as' after import module" in error_lower:
        return 'import without as (import file("x"))'
    
    if 'end, found: as' in error_lower:
        return 'provide as (provide x as y)'
    
    if 'end, found: dot' in error_lower:
        return 'provide from (provide from M: x end)'
    
    # "Unexpected tokens after program end" - check the actual file
    if 'unexpected tokens after program end' in error_lower:
        match = re.search(r'location:\s*"(\d+):(\d+):(\d+)"', error)
        if match:
            line_num = int(match.group(1))
            
            # Check file content for specific patterns
            patterns = {
                'type alias (type Name = Type)': r'^type\s+\w+\s*[=<]',
                'newtype (newtype Name: ...)': r'^newtype\s+\w+:',
                'cases block (cases(...) x block:)': r'cases\s*\([^)]+\)\s+\w+\s+block:',
                'table literal (table: col row: val end)': r'^\s*table:',
                'lambda block (lam(...) block:)': r'lam\s*\([^)]*\)\s+block:',
                '#lang directive': r'^#lang\s+',
                'provide-types': r'^provide-types',
                'ask expression (ask: | cond then: body end)': r'^\s*ask:',
            }
            
            feature = check_file_content(filename, line_num, patterns)
            if feature:
                return feature
            
            # Fallback based on location
            if line_num <= 3:
                return '#lang or provide-types (prelude)'
            elif line_num <= 15:
                return 'Advanced import/provide/type'
            else:
                return 'Unknown feature (line {})'.format(line_num)
        return 'Unknown parse error'
    
    return 'Unknown error type'

# Process all files
output_lines = []
for filename, error in files:
    category = categorize_error(filename, error)
    output_lines.append(f"{filename}  # {category}")

# Write annotated output
with open('bulk_test_results/failing_files.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

# Print summary
categories = [categorize_error(f, e) for f, e in files]
summary = Counter(categories)

print("=" * 90)
print("📊 MISSING FEATURE ANALYSIS - COMPLETE BREAKDOWN")
print("=" * 90)
print()

# Group by category
prelude = sum(v for k,v in summary.items() if 'lang' in k.lower() or 'prelude' in k.lower())
imports = sum(v for k,v in summary.items() if 'import' in k.lower() or 'provide' in k.lower())
types = sum(v for k,v in summary.items() if 'type alias' in k.lower() or 'newtype' in k.lower())
runtime = sum(v for k,v in summary.items() if 'cases block' in k.lower() or 'table' in k.lower() or 'ask' in k.lower() or 'lambda block' in k.lower())
success = summary.get('SUCCESS - Parses correctly now!', 0)
unknown = sum(v for k,v in summary.items() if 'unknown' in k.lower())

print(f"✅ SUCCESS (already parsing):          {success:3d}")
print(f"📝 Type System Features:               {types:3d}")
print(f"   - type alias (type Name = Type)     {summary.get('type alias (type Name = Type)', 0):3d}")
print(f"   - newtype declarations              {summary.get('newtype (newtype Name: ...)', 0):3d}")
print()
print(f"🔧 Import/Export Features:             {imports:3d}")
print(f"   - import from syntax                {summary.get('import from (import x, y from file(...))', 0):3d}")
print(f"   - provide block syntax              {summary.get('provide block (provide { x: x, f: f } end)', 0):3d}")
print(f"   - type in provide                   {summary.get('type in provide (provide: type T, data D end)', 0):3d}")
print(f"   - data in provide                   {summary.get('data in provide (provide: data D end)', 0):3d}")
print(f"   - asterisk in provide               {summary.get('asterisk in provide (provide: *, type T end)', 0):3d}")
print(f"   - provide as syntax                 {summary.get('provide as (provide x as y)', 0):3d}")
print(f"   - provide from syntax               {summary.get('provide from (provide from M: x end)', 0):3d}")
print(f"   - import without as                 {summary.get('import without as (import file("x"))', 0):3d}")
print(f"   - module in provide                 {summary.get('module in provide (provide: module M end)', 0):3d}")
print()
print(f"🎯 Runtime Features:                   {runtime:3d}")
print(f"   - cases block                       {summary.get('cases block (cases(...) x block:)', 0):3d}")
print(f"   - table literals                    {summary.get('table literal (table: col row: val end)', 0):3d}")
print(f"   - ask expressions                   {summary.get('ask expression (ask: | cond then: body end)', 0):3d}")
print(f"   - lambda block                      {summary.get('lambda block (lam(...) block:)', 0):3d}")
print()
print(f"🏁 Prelude Features:                   {prelude:3d}")
print(f"   - #lang directive                   {summary.get('#lang directive', 0):3d}")
print(f"   - provide-types                     {summary.get('provide-types', 0):3d}")
print(f"   - #lang or provide-types            {summary.get('#lang or provide-types (prelude)', 0):3d}")
print()
print(f"❓ Unknown/Mixed:                      {unknown:3d}")
print("=" * 90)
print(f"\n📈 TOTAL FILES: {len(files)}")
print(f"✅ Working: {success} ({100*success//len(files)}%)")
print(f"🔴 Failing: {len(files) - success} ({100*(len(files)-success)//len(files)}%)")
