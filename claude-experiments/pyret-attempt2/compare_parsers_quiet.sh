#!/bin/bash
# Compare Pyret's official parser with our Rust parser (quiet version for automated testing)
# Usage: ./compare_parsers_quiet.sh "2 + 3" [unique_id]

if [ -z "$1" ]; then
    exit 1
fi

EXPR="$1"
UNIQUE_ID="${2:-default}"

TEMP_FILE="/tmp/pyret_compare_input_${UNIQUE_ID}.arr"
PYRET_JSON="/tmp/pyret_output_${UNIQUE_ID}.json"
RUST_JSON="/tmp/rust_output_${UNIQUE_ID}.json"
PYRET_EXPR="/tmp/pyret_expr_${UNIQUE_ID}.json"

# Write expression to temp file
echo "$EXPR" > "$TEMP_FILE"

# Parse with Pyret's official parser
cd /Users/jimmyhmiller/Documents/Code/open-source/pyret-lang
node ast-to-json.jarr "$TEMP_FILE" "$PYRET_JSON" 2>&1 > /dev/null || { rm -f "$TEMP_FILE" "$PYRET_JSON"; exit 1; }

# Extract just the expression from Pyret's output (first statement in body)
python3 -c "
import json, sys
try:
    with open('$PYRET_JSON') as f:
        data = json.load(f)
    if 'body' in data and 'stmts' in data['body'] and len(data['body']['stmts']) > 0:
        with open('$PYRET_EXPR', 'w') as out:
            json.dump(data['body']['stmts'][0], out, indent=2)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" || { rm -f "$TEMP_FILE" "$PYRET_JSON" "$RUST_JSON" "$PYRET_EXPR"; exit 1; }

# Parse with our Rust parser
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2
./target/debug/to_pyret_json "$TEMP_FILE" 2>/dev/null > "$RUST_JSON" || { rm -f "$TEMP_FILE" "$PYRET_JSON" "$RUST_JSON" "$PYRET_EXPR"; exit 1; }

# Compare the two JSON outputs (normalize for field order)
python3 << EOF
import json
import sys

def normalize_json(obj):
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    else:
        return obj

try:
    with open('$PYRET_EXPR') as f:
        pyret = json.load(f)
    with open('$RUST_JSON') as f:
        rust = json.load(f)

    pyret_norm = normalize_json(pyret)
    rust_norm = normalize_json(rust)

    if pyret_norm == rust_norm:
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
EOF

RESULT=$?

# Clean up temp files
rm -f "$TEMP_FILE" "$PYRET_JSON" "$RUST_JSON" "$PYRET_EXPR"

exit $RESULT
