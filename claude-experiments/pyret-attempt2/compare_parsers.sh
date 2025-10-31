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

# Extract just the expression from Pyret's output (first statement in body)
python3 -c "
import json, sys
with open('$PYRET_JSON') as f:
    data = json.load(f)
if 'body' in data and 'stmts' in data['body'] and len(data['body']['stmts']) > 0:
    with open('$PYRET_EXPR', 'w') as out:
        json.dump(data['body']['stmts'][0], out, indent=2)
else:
    print('ERROR: No expression found in Pyret output', file=sys.stderr)
    sys.exit(1)
"

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

def normalize_json(obj):
    """Recursively sort dictionaries for consistent comparison"""
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
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
