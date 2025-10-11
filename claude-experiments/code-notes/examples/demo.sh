#!/bin/bash
# Demonstration script for code-notes
# This shows how to use the various commands

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
BINARY="$PROJECT_ROOT/target/release/code-notes"

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Building code-notes..."
    cd "$PROJECT_ROOT"
    cargo build --release
fi

# Always run from the examples directory so relative paths work
cd "$SCRIPT_DIR"

echo "=== Code Notes Demo ==="
echo

# Initialize
echo "1. Initializing code-notes in current directory..."
$BINARY init
echo

# Create a collection
echo "2. Creating an 'onboarding' collection..."
$BINARY create-collection "onboarding" --description "Notes for new team members"
echo

# Add some notes to the sample code
echo "3. Adding notes to sample_code.rs..."

# Note on authenticate_user function
$BINARY add \
    --file sample_code.rs \
    --line 8 \
    --column 8 \
    --content "Main entry point for authentication. Uses token-based auth. See auth-design.md for details." \
    --author "Alice" \
    --collection onboarding

# Note on load_user_database
$BINARY add \
    --file sample_code.rs \
    --line 23 \
    --column 4 \
    --content "Currently loads from in-memory HashMap. TODO: Replace with actual database in production." \
    --author "Bob" \
    --collection onboarding

# Note on password hashing
$BINARY add \
    --file sample_code.rs \
    --line 43 \
    --column 4 \
    --content "SECURITY: This is simplified for demo. Production uses bcrypt with proper salting." \
    --author "SecurityTeam" \
    --collection onboarding

echo "Added 3 notes"
echo

# List all notes
echo "4. Listing all notes in onboarding collection..."
$BINARY list --collection onboarding
echo

# List notes for specific file
echo "5. Listing notes for sample_code.rs..."
$BINARY list sample_code.rs --collection onboarding
echo

# Show collections
echo "6. Showing all collections..."
$BINARY collections
echo

# Export collection
echo "7. Exporting onboarding collection..."
$BINARY export onboarding --output onboarding-notes.json
echo "Exported to onboarding-notes.json"
echo

echo "=== Demo Complete ==="
echo
echo "Notes are stored globally in ~/.code-notes/"
echo
echo "You can now:"
echo "  - View individual notes with: $BINARY view <note-id>"
echo "  - Update notes with: $BINARY update <note-id> --content \"new content\""
echo "  - Delete notes with: $BINARY delete <note-id> --collection onboarding"
echo "  - Migrate notes after commits with: $BINARY migrate --collection onboarding"
echo
echo "Check your notes directory:"
echo "  ls ~/.code-notes/"
