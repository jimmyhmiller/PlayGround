#!/bin/bash

# Test what happens when a function body completely changes

set -e

PROJECT_DIR="/tmp/code-notes-body-test-$(date +%s)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_NOTES="$SCRIPT_DIR/../target/debug/code-notes"

echo "=== Testing Complete Function Body Change ==="
echo ""

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
git init

# Create initial function
cat > math.rs <<'EOF'
fn calculate_tax(income: f64) -> f64 {
    income * 0.2
}
EOF

git add math.rs
git commit -m "Initial simple tax calculation"

echo "Created initial function with simple body"
cat math.rs
echo ""

# Initialize and add note to the FUNCTION (not just signature)
$CODE_NOTES init
$CODE_NOTES add \
    --file math.rs \
    --line 1 \
    --column 0 \
    --content "This calculates income tax - keep this updated with latest tax laws" \
    --author "Tax Expert"

echo "Added note to function"
echo ""

# Completely change the body - complex implementation
cat > math.rs <<'EOF'
fn calculate_tax(income: f64) -> f64 {
    // New progressive tax system
    let brackets = vec![
        (10000.0, 0.1),
        (40000.0, 0.2),
        (100000.0, 0.3),
    ];

    let mut tax = 0.0;
    let mut remaining = income;
    let mut prev_limit = 0.0;

    for (limit, rate) in brackets {
        let taxable = (remaining.min(limit - prev_limit)).max(0.0);
        tax += taxable * rate;
        remaining -= taxable;
        prev_limit = limit;
        if remaining <= 0.0 {
            break;
        }
    }

    tax + remaining * 0.4
}
EOF

git add math.rs
git commit -m "Complete rewrite - progressive tax brackets"

echo "Completely rewrote function body:"
cat math.rs
echo ""

# Migrate
echo "Running migration..."
$CODE_NOTES migrate

echo ""
echo "Checking results..."
$CODE_NOTES list math.rs

echo ""
echo "=== Detailed view ==="
NOTE_ID=$($CODE_NOTES list math.rs | grep "^ID:" | awk '{print $2}')
$CODE_NOTES view "$NOTE_ID"
