#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

echo "Let me show you EXACTLY how categorization works with real examples:"
echo ""
echo "=" $(printf '=%.0s' {1..70})

# Example 1: Import from syntax
echo ""
echo "EXAMPLE 1: import from syntax"
echo "File: tests/pyret/tests/modules/alias-x.arr"
echo ""
echo "Step 1 - What error does the parser give?"
cargo run --quiet --bin to_pyret_json "$BASE_DIR/tests/pyret/tests/modules/alias-x.arr" 2>&1 | grep "Error" | head -1
echo ""
echo "Step 2 - What's in the file at that location?"
head -2 "$BASE_DIR/tests/pyret/tests/modules/alias-x.arr"
echo ""
echo "Step 3 - How does the script categorize it?"
echo "The error contains 'Colon, found: From' so it's categorized as 'import from'"
echo ""

echo "=" $(printf '=%.0s' {1..70})

# Example 2: Type alias
echo ""
echo "EXAMPLE 2: type alias"
echo "File: tools/benchmark/auto-report-programs/anf-loop-compiler.arr"
echo ""
echo "Step 1 - What error does the parser give?"
cargo run --quiet --bin to_pyret_json "$BASE_DIR/tools/benchmark/auto-report-programs/anf-loop-compiler.arr" 2>&1 | grep "Error" | head -1
echo ""
echo "Step 2 - What's on line 22 (the error location)?"
sed -n '22p' "$BASE_DIR/tools/benchmark/auto-report-programs/anf-loop-compiler.arr"
echo ""
echo "Step 3 - How does the script categorize it?"
echo "Script sees 'Unexpected tokens after program end' at line 22"
echo "It reads line 22 and finds: 'type Loc = SL.Srcloc'"
echo "The regex '^type\s+\w+\s*[=<]' matches, so it's categorized as 'type alias'"
echo ""

echo "=" $(printf '=%.0s' {1..70})

# Example 3: Provide block
echo ""
echo "EXAMPLE 3: provide block syntax"
echo "File: tests/pyret/tests/exporter.arr"
echo ""
echo "Step 1 - What error does the parser give?"
cargo run --quiet --bin to_pyret_json "$BASE_DIR/tests/pyret/tests/exporter.arr" 2>&1 | grep "Error" | head -1
echo ""
echo "Step 2 - What's in the file at that location?"
head -3 "$BASE_DIR/tests/pyret/tests/exporter.arr"
echo ""
echo "Step 3 - How does the script categorize it?"
echo "The error contains 'Colon, found: LBrace' so it's categorized as 'provide block'"
echo ""

echo "=" $(printf '=%.0s' {1..70})

# Example 4: Advanced categorization
echo ""
echo "EXAMPLE 4: 'advanced import/provide/type' catch-all"
echo "File: pyret-ast.arr"
echo ""
echo "Step 1 - What error does the parser give?"
cargo run --quiet --bin to_pyret_json "$BASE_DIR/pyret-ast.arr" 2>&1 | grep "Error" | head -1
echo ""
echo "Step 2 - What's on line 10?"
sed -n '10p' "$BASE_DIR/pyret-ast.arr"
echo ""
echo "Step 3 - How does the script categorize it?"
echo "Script sees 'Unexpected tokens' at line 10"
echo "Line 10 doesn't match specific patterns (type alias, cases block, etc.)"
echo "Line 10 is between 4-15, so it's in the import/provide region"
echo "Gets categorized as 'advanced import/provide/type'"
echo ""

echo "=" $(printf '=%.0s' {1..70})
echo ""
echo "SUMMARY:"
echo "  1. Specific error messages → specific categories (import from, provide block, etc.)"
echo "  2. Generic 'Unexpected tokens' → reads the actual file line"
echo "  3. Regex patterns match syntax (type alias, cases block, etc.)"
echo "  4. Fallback based on line number location (prelude, imports, runtime)"
