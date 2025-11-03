#!/bin/bash
# Compare Pyret's official parser with our Rust parser
# Usage: ./compare_parsers.sh "2 + 3"

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 '<pyret expression>'"
    echo "Example: $0 '2 + 3'"
    exit 1
fi

EXPR="$1"
TEMP_FILE="/tmp/pyret_compare_input.arr"
PYRET_JSON="/tmp/pyret_output.json"
RUST_JSON="/tmp/rust_output.json"
PYRET_EXPR="/tmp/pyret_expr.json"

# Write expression to temp file
echo "$EXPR" > "$TEMP_FILE"

echo "=== Input ==="
echo "$EXPR"
echo

# Parse with Pyret's official parser
echo "=== Pyret Parser ==="
cd /Users/jimmyhmiller/Documents/Code/open-source/pyret-lang
node ast-to-json.jarr "$TEMP_FILE" "$PYRET_JSON" 2>&1 | grep "JSON written" || true

# Copy the full program AST (no longer extracting just the first statement)
cp "$PYRET_JSON" "$PYRET_EXPR"

cat "$PYRET_EXPR"
echo
echo

# Parse with our Rust parser
echo "=== Rust Parser ==="
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2
cargo run --bin to_pyret_json "$TEMP_FILE" 2>/dev/null > "$RUST_JSON"
cat "$RUST_JSON"
echo
echo

# Compare the two JSON outputs (normalize for field order)
echo "=== Comparison ==="
python3 << 'EOF'
import json
import sys
import re

def normalize_srcloc(text):
    """Normalize srcloc strings to ignore filename differences"""
    if isinstance(text, str):
        # Replace any filename in srcloc(...) with "file.arr"
        return re.sub(r'srcloc\("([^"]+)"', 'srcloc("file.arr"', text)
    return text

def normalize_json(obj):
    """Recursively sort dictionaries and normalize srcloc strings"""
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    elif isinstance(obj, str):
        return normalize_srcloc(obj)
    else:
        return obj

with open('/tmp/pyret_expr.json') as f:
    pyret = json.load(f)

with open('/tmp/rust_output.json') as f:
    rust = json.load(f)

pyret_norm = normalize_json(pyret)
rust_norm = normalize_json(rust)

if pyret_norm == rust_norm:
    print("✅ IDENTICAL - Parsers produce the same AST!")
    sys.exit(0)
else:
    print("❌ DIFFERENT - Found differences:")
    print()
    print("Pyret AST:")
    print(json.dumps(pyret_norm, indent=2))
    print()
    print("Rust AST:")
    print(json.dumps(rust_norm, indent=2))
    sys.exit(1)
EOF
