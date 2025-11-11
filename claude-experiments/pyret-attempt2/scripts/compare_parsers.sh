#!/bin/bash
# Compare Pyret's official parser with our Rust parser
# Usage: ./compare_parsers.sh "2 + 3"
#    OR: ./compare_parsers.sh /path/to/file.arr

# Note: Don't use set -e because we want to see the comparison output
# even when parsers differ (Python script exits with code 1)

if [ -z "$1" ]; then
    echo "Usage: $0 '<pyret expression>' OR $0 '/path/to/file.arr'"
    echo "Example: $0 '2 + 3'"
    echo "Example: $0 tests/pyret/tests/test-strings.arr"
    exit 1
fi

INPUT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEMP_FILE="/tmp/pyret_compare_input.arr"
PYRET_JSON="/tmp/pyret_output.json"
RUST_JSON="/tmp/rust_output.json"
PYRET_EXPR="/tmp/pyret_expr.json"

# Check if PYRET_REPO environment variable is set
if [ -z "$PYRET_REPO" ]; then
    echo "ERROR: PYRET_REPO environment variable is not set."
    echo ""
    echo "Usage: PYRET_REPO=/path/to/pyret-lang $0 '<expression>'"
    echo "   OR: PYRET_REPO=/path/to/pyret-lang $0 '/path/to/file.arr'"
    echo ""
    echo "Or set it permanently in your shell profile:"
    echo "  export PYRET_REPO=/path/to/pyret-lang"
    echo ""
    exit 1
fi

# Validate that PYRET_REPO points to a valid pyret-lang repository
if [ ! -f "$PYRET_REPO/ast-to-json.jarr" ]; then
    echo "ERROR: $PYRET_REPO does not contain ast-to-json.jarr"
    echo "Please ensure PYRET_REPO points to a valid pyret-lang repository."
    exit 1
fi

# Check if input is a file path or an expression
if [ -f "$INPUT" ]; then
    # It's a file - convert to absolute path for use across directories
    TEMP_FILE="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
    echo "=== Input File ==="
    echo "$INPUT"
    echo "(File contains $(wc -l < "$INPUT") lines)"
else
    # It's an expression - write to temp file
    echo "$INPUT" > "$TEMP_FILE"
    echo "=== Input ==="
    echo "$INPUT"
fi
echo

# Parse with Pyret's official parser
echo "=== Parsing with Pyret's official parser... ==="
cd "$PYRET_REPO"
node ast-to-json.jarr "$TEMP_FILE" "$PYRET_JSON" 2>&1 | grep "JSON written" || true

# Copy the full program AST (no longer extracting just the first statement)
cp "$PYRET_JSON" "$PYRET_EXPR"

# Parse with our Rust parser
echo "=== Parsing with Rust parser... ==="
cd "$PROJECT_ROOT"
RUST_ERROR=$(mktemp)
if cargo run --bin to_pyret_json "$TEMP_FILE" > "$RUST_JSON" 2>"$RUST_ERROR"; then
    RUST_PARSE_SUCCESS=1
else
    RUST_PARSE_SUCCESS=0
    echo "❌ RUST PARSER ERROR:"
    cat "$RUST_ERROR"
    rm "$RUST_ERROR"
    exit 1
fi
rm "$RUST_ERROR"

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

try:
    with open('/tmp/pyret_expr.json') as f:
        pyret = json.load(f)
except Exception as e:
    print(f"Error loading Pyret JSON: {e}")
    sys.exit(1)

try:
    with open('/tmp/rust_output.json') as f:
        rust = json.load(f)
except Exception as e:
    print(f"Error loading Rust JSON: {e}")
    sys.exit(1)

pyret_norm = normalize_json(pyret)
rust_norm = normalize_json(rust)

if pyret_norm == rust_norm:
    print("✅ IDENTICAL - Parsers produce the same AST!")
    sys.exit(0)
else:
    print("❌ DIFFERENT - Found differences:")
    print()

    # Save full ASTs to files for diffing
    with open('/tmp/pyret_ast_normalized.json', 'w') as f:
        json.dump(pyret_norm, f, indent=2)

    with open('/tmp/rust_ast_normalized.json', 'w') as f:
        json.dump(rust_norm, f, indent=2)

    sys.exit(1)
EOF

# Capture the exit code from Python
EXIT_CODE=$?

# If different, show the diff in git-style unified format
if [ $EXIT_CODE -eq 1 ]; then
    echo
    echo "=== Diff (Pyret official vs Rust implementation) ==="
    # Use git diff with --no-index for colored, unified diff output
    # or fall back to regular diff -u if git is not available
    if command -v git &> /dev/null; then
        git diff --no-index --color=always /tmp/pyret_ast_normalized.json /tmp/rust_ast_normalized.json || true
    else
        diff -u /tmp/pyret_ast_normalized.json /tmp/rust_ast_normalized.json || true
    fi
fi

exit $EXIT_CODE
