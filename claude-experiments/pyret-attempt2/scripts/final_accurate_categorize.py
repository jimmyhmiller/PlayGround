#!/usr/bin/env python3
import re
import os
import subprocess
from collections import Counter

BASE_DIR = "/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

def can_parse(filename):
    """Test if our parser can actually parse the file"""
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        return False, "file not found"
    
    try:
        result = subprocess.run(
            ["cargo", "run", "--quiet", "--bin", "to_pyret_json", full_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            # Check for specific errors
            if "stack overflow" in result.stderr.lower():
                return False, "stack overflow (expression too deep)"
            elif "abort" in result.stderr.lower() or "signal" in result.stderr.lower():
                return False, "crash/abort"
            else:
                return False, "parse error"
        return True, "success"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"error: {e}"

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

def check_file_content(filename, line_num):
    """Check what's actually on the error line"""
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        return None
    
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
            if 0 <= line_num-1 < len(lines):
                line = lines[line_num-1].strip()
                
                # Check for specific patterns we know
                if re.match(r'^type\s+\w+\s*=', line):
                    return 'type alias (type Name = Type)'
                if re.match(r'^type\s+\w+\s*<', line):
                    return 'type alias with generics (type Name<T> = ...)'
                if re.match(r'^newtype\s+\w+:', line):
                    return 'newtype (newtype Name: ...)'
                if re.search(r'cases\s*\([^)]+\)\s+\w+\s+block:', line):
                    return 'cases block (cases(T) x block: ...)'
                if re.match(r'^\s*table:', line):
                    return 'table literal (table: col row: val end)'
                if re.search(r'lam\s*\([^)]*\)\s+block:', line):
                    return 'lambda block (lam(...) block: ... end)'
                if re.match(r'^#lang\s+', line):
                    return '#lang directive'
                if re.match(r'^provide-types', line):
                    return 'provide-types statement'
                if re.match(r'^\s*ask:', line):
                    return 'ask expression (ask: | cond then: body end)'
                
                # Return the actual line content for unknown cases
                return f"UNKNOWN: line={line_num} content='{line[:60]}...'"
    except:
        pass
    return None

def categorize_error(filename, error):
    """Determine which missing feature causes the error"""
    
    # For "SUCCESS" cases, actually test if they parse
    if error.startswith('SUCCESS'):
        can_parse_file, reason = can_parse(filename)
        if can_parse_file:
            return 'SUCCESS (verified)'
        else:
            return f'FAIL despite earlier success ({reason})'
    
    error_lower = error.lower()
    
    # Specific error patterns from tokenizer/parser - these are reliable
    if 'colon, found: from' in error_lower:
        return 'import from (import x, y from file(...))'
    
    if 'colon, found: lbrace' in error_lower:
        return 'provide block (provide { x: x } end)'
    
    if 'name, found: type' in error_lower:
        return 'type in provide (provide: type T end)'
    
    if 'name, found: times' in error_lower:
        return 'asterisk in provide (provide: * end)'
    
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
    
    # Generic "Unexpected tokens" - check what's actually there
    if 'unexpected tokens after program end' in error_lower:
        match = re.search(r'location:\s*"(\d+):(\d+):(\d+)"', error)
        if match:
            line_num = int(match.group(1))
            feature = check_file_content(filename, line_num)
            if feature:
                return feature
        return 'UNKNOWN: unexpected tokens (no line number)'
    
    # Don't know what this is
    return f'UNKNOWN: {error[:80]}'

print("Testing all files... (this may take a minute)")
print("")

# Process all files
output_lines = []
for i, (filename, error) in enumerate(files):
    if i % 50 == 0:
        print(f"Processing {i}/{len(files)}...")
    category = categorize_error(filename, error)
    output_lines.append(f"{filename}  # {category}")

# Write annotated output
with open('bulk_test_results/failing_files.txt', 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

# Print summary
categories = [categorize_error(f, e) for f, e in files]
summary = Counter(categories)

print("\n" + "=" * 90)
print("📊 ACCURATE MISSING FEATURE ANALYSIS")
print("=" * 90)
print()

success = sum(v for k,v in summary.items() if 'SUCCESS' in k and 'verified' in k.lower())
crash = sum(v for k,v in summary.items() if 'stack overflow' in k.lower() or 'crash' in k.lower())
fail_despite = sum(v for k,v in summary.items() if 'FAIL despite' in k)
unknown = sum(v for k,v in summary.items() if 'UNKNOWN' in k)

print(f"✅ Actually parsing correctly:         {success:3d}")
print(f"💥 Stack overflow / crashes:           {crash:3d}")
print(f"❌ False positives (don't really work): {fail_despite:3d}")
print(f"❓ Unknown (need manual investigation): {unknown:3d}")
print()

# Known features
known = sum(v for k,v in summary.items() if 'UNKNOWN' not in k and 'SUCCESS' not in k and 'FAIL despite' not in k)
print(f"🔍 Identified missing features:        {known:3d}")
print()

# Show top categories
print("Top categories:")
for cat, count in sorted(summary.items(), key=lambda x: -x[1])[:15]:
    if 'UNKNOWN' not in cat:
        print(f"  {count:3d}  {cat}")

print()
print("=" * 90)
print(f"TOTAL: {len(files)} files")
print(f"Actually working: {success}/{len(files)} ({100*success//len(files)}%)")
print(f"Need investigation: {unknown} files")
