#!/bin/bash

# Test script to verify bug fixes for migration issues

set -e

PROJECT_DIR="/tmp/code-notes-test-$(date +%s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_NOTES="$SCRIPT_DIR/../target/debug/code-notes"

echo "=== Testing Migration Bug Fixes ==="
echo ""

# Create test project
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
git init

echo "Test project created at: $PROJECT_DIR"
echo ""

# ============================================================================
# TEST 1: Whitespace-only changes (Bug: light-flustered-bat)
# ============================================================================

echo "=== TEST 1: Whitespace-only changes ==="
echo ""

# Create initial file with unformatted code
cat > sample.rs <<'EOF'
fn calculate(x: i32) -> i32 {
x * 2
}
EOF

git add sample.rs
git commit -m "Initial commit with unformatted code"

echo "Created sample.rs with unformatted code"
echo ""

# Initialize code-notes
$CODE_NOTES init

# Add a note to the function (lines are 1-indexed)
$CODE_NOTES add \
    --file sample.rs \
    --line 1 \
    --column 0 \
    --content "This function doubles the input value" \
    --author "Test User"

echo "Added note to function"
echo ""

# Format the code (only whitespace change)
cat > sample.rs <<'EOF'
fn calculate(x: i32) -> i32 {
    x * 2
}
EOF

git add sample.rs
git commit -m "Format code (whitespace only)"

echo "Formatted code - only whitespace changed"
echo ""

# Migrate notes
echo "Running migration..."
$CODE_NOTES migrate

echo ""
echo "Checking if note survived whitespace-only change..."
echo "Listing notes for sample.rs:"
$CODE_NOTES list sample.rs
echo ""
NOTE_COUNT=$($CODE_NOTES list sample.rs | grep -c "This function doubles" || true)

if [ "$NOTE_COUNT" -eq 1 ]; then
    echo "✓ TEST 1 PASSED: Note survived whitespace-only change"
else
    echo "✗ TEST 1 FAILED: Note was orphaned by whitespace change"
    exit 1
fi

echo ""

# ============================================================================
# TEST 2: Function body changes with unchanged signature (Bug: only-muffled-kangaroo)
# ============================================================================

echo "=== TEST 2: Function body changes with unchanged signature ==="
echo ""

# Create file with simple function
cat > math.rs <<'EOF'
fn add(x: i32, y: i32) -> i32 {
    x + y
}
EOF

git add math.rs
git commit -m "Add simple add function"

echo "Created math.rs with simple function"
echo ""

# Add note to function signature (lines are 1-indexed)
$CODE_NOTES add \
    --file math.rs \
    --line 1 \
    --column 0 \
    --content "Addition function for two integers" \
    --author "Test User"

echo "Added note to function signature"
echo ""

# Refactor function body (signature unchanged)
cat > math.rs <<'EOF'
fn add(x: i32, y: i32) -> i32 {
    // New implementation with logging
    let result = x + y;
    println!("Adding {} + {} = {}", x, y, result);
    result
}
EOF

git add math.rs
git commit -m "Refactor function body with logging"

echo "Refactored function body (signature unchanged)"
echo ""

# Migrate notes
echo "Running migration..."
$CODE_NOTES migrate

echo ""
echo "Checking if note survived function body change..."
echo "Listing notes for math.rs:"
$CODE_NOTES list math.rs
echo ""
NOTE_COUNT=$($CODE_NOTES list math.rs | grep -c "Addition function" || true)

if [ "$NOTE_COUNT" -eq 1 ]; then
    echo "✓ TEST 2 PASSED: Note survived function body change"
else
    echo "✗ TEST 2 FAILED: Note was orphaned by function body change"
    exit 1
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=== ALL TESTS PASSED ==="
echo ""
echo "✓ Whitespace-only changes no longer orphan notes"
echo "✓ Function body changes no longer orphan signature notes"
echo ""
echo "Test project location: $PROJECT_DIR"
echo "(You can delete it manually when done)"
