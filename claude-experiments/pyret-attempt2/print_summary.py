#!/usr/bin/env python3
from collections import Counter

# Read the annotated file
with open('bulk_test_results/failing_files.txt', 'r') as f:
    lines = f.readlines()

categories = []
for line in lines:
    if '#' in line:
        category = line.split('#')[1].strip()
        categories.append(category)

summary = Counter(categories)

print("=" * 90)
print("📊 PARSER STATUS - HONEST BREAKDOWN")
print("=" * 90)
print()

# Success
success_count = 0
for cat, count in sorted(summary.items()):
    if 'SUCCESS (verified)' in cat:
        print(f"✅ WORKING ({count} files):")
        print(f"   {count:3d}  Files produce identical ASTs to official parser")
        success_count = count
        print()

# AST mismatches
print("🔸 PARSE BUT HAVE AST ISSUES:")
for cat, count in sorted(summary.items()):
    if any(x in cat for x in ['underscore', 'dot/bang', 'other AST']):
        print(f"   {count:3d}  {cat}")
print()

# Known missing features
print("❌ IDENTIFIED MISSING FEATURES:")
known_features = []
for cat, count in sorted(summary.items(), key=lambda x: -x[1]):
    if 'UNKNOWN' not in cat and 'SUCCESS' not in cat and 'FAIL despite' not in cat and 'underscore' not in cat and 'dot/bang' not in cat and 'other AST' not in cat:
        known_features.append((cat, count))

for cat, count in known_features[:20]:  # Show top 20
    print(f"   {count:3d}  {cat}")
print()

# Unknowns
print("❓ NEED MANUAL INVESTIGATION:")
unknown_count = 0
for cat, count in sorted(summary.items()):
    if 'UNKNOWN' in cat:
        unknown_count += count

if unknown_count > 0:
    print(f"   {unknown_count:3d}  files with unknown issues")
    print()
    print("   To see them:")
    print('   grep "UNKNOWN:" bulk_test_results/failing_files.txt | head -20')
print()

print("=" * 90)
total = len(categories)
print(f"TOTAL: {success_count} / {total} files working ({100*success_count//total if total > 0 else 0}%)")
print(f"Known missing features: {sum(c for _, c in known_features)} files")
print(f"Need investigation: {unknown_count} files")
print()
print("Run './reannotate.sh' after making parser changes")
print("=" * 90)
