#!/usr/bin/env python3
"""
Find files where gix-blame and git blame differ
"""
import json
import subprocess
import sys

# Load the survival data
with open('/tmp/test-gix-new-full/survival.json') as f:
    survival = json.load(f)

with open('/tmp/bench-python-playground/survival.json') as f:
    python_survival = json.load(f)

print("Looking for files with significant discrepancies...")
print()

discrepancies = []
for file_hash in survival:
    if file_hash not in python_survival:
        continue

    rust_lines = survival[file_hash][-1][1] if survival[file_hash] else 0
    python_lines = python_survival[file_hash][-1][1] if python_survival[file_hash] else 0

    if rust_lines != python_lines and rust_lines > 0 and python_lines > 0:
        diff = python_lines - rust_lines
        pct = (diff / python_lines * 100) if python_lines > 0 else 0
        discrepancies.append((file_hash, rust_lines, python_lines, diff, pct))

# Sort by percentage difference
discrepancies.sort(key=lambda x: abs(x[4]), reverse=True)

print(f"Found {len(discrepancies)} files with discrepancies")
print()
print("Top 20 files with largest percentage differences:")
print("-" * 100)
print(f"{'Hash':<45} {'Rust':<10} {'Python':<10} {'Diff':<10} {'%':<8}")
print("-" * 100)

for file_hash, rust_lines, python_lines, diff, pct in discrepancies[:20]:
    print(f"{file_hash:<45} {rust_lines:<10} {python_lines:<10} {diff:<10} {pct:>6.1f}%")

# Now let's look at absolute differences
print()
print("Top 20 files with largest absolute differences:")
print("-" * 100)
print(f"{'Hash':<45} {'Rust':<10} {'Python':<10} {'Diff':<10} {'%':<8}")
print("-" * 100)

discrepancies.sort(key=lambda x: abs(x[3]), reverse=True)
for file_hash, rust_lines, python_lines, diff, pct in discrepancies[:20]:
    print(f"{file_hash:<45} {rust_lines:<10} {python_lines:<10} {diff:<10} {pct:>6.1f}%")
