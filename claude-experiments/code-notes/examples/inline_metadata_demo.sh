#!/bin/bash

# Inline Metadata Feature Demonstration
# Shows how to use @meta: in inline comments

set -e

echo "=================================================="
echo "Code Notes - Inline Metadata Demo"
echo "=================================================="
echo ""

# Create a temporary demo directory
DEMO_DIR=$(mktemp -d -t code-notes-inline-meta-demo)
cd "$DEMO_DIR"

echo "Demo directory: $DEMO_DIR"
echo ""

# Initialize git repo
echo "1. Initializing git repository..."
git init
git config user.name "Demo User"
git config user.email "demo@example.com"

# Initialize code-notes
code-notes init

echo ""
echo "=================================================="
echo "Example 1: Basic @meta: syntax"
echo "=================================================="
echo ""

# Create a file with @note: and @meta:
echo "2. Creating file with @note: and @meta: markers..."
cat > security.rs << 'EOF'
// @note: This authentication function has a security vulnerability
// @meta: {"tags":["bug","security"],"severity":"critical","priority":9}
pub fn authenticate(username: &str, password: &str) -> bool {
    // Insecure: plaintext password comparison
    username == "admin" && password == "secret"
}

// @note: Authorization check is too permissive
// @meta: {"tags":["security","enhancement"],"priority":7}
pub fn authorize(user: &str, resource: &str) -> bool {
    user == "admin"  // Everyone is admin!
}

// @note: Password hashing is needed here
// @meta: {"tags":["enhancement"],"priority":8,"linked_issues":["SEC-456"]}
pub fn hash_password(password: &str) -> String {
    // TODO: Implement bcrypt
    password.to_string()
}
EOF

git add security.rs
git commit -m "Add security module with notes"

echo ""
echo "3. Capturing notes with inline metadata..."
code-notes capture security.rs \
  --author "Security Team" \
  --collection security

echo ""
echo "=================================================="
echo "Example 2: Mixing CLI and inline metadata"
echo "=================================================="
echo ""

# Create another file
cat > config.rs << 'EOF'
// @note: Configuration should support environment-based overrides
// @meta: {"priority":6,"tags":["enhancement"]}
pub struct Config {
    pub database_url: String,
    pub api_key: String,
}

// @note: This needs validation
// No @meta: line here - will use CLI defaults
pub fn load_config() -> Config {
    Config {
        database_url: "postgres://localhost".to_string(),
        api_key: "changeme".to_string(),
    }
}
EOF

git add config.rs
git commit -m "Add config module"

echo "4. Capturing with CLI metadata as default..."
code-notes capture config.rs \
  --author "DevOps Team" \
  --collection config \
  --metadata '{"audience":"all-devs","trail_id":"config-setup"}'

echo ""
echo "=================================================="
echo "Example 3: Multiline notes with metadata"
echo "=================================================="
echo ""

cat > api.rs << 'EOF'
// @note: Main API endpoint handler
// This handles all incoming requests
// Needs rate limiting and authentication
// @meta: {"tags":["api","enhancement"],"priority":5,"estimated_hours":8}
pub async fn handle_request(req: Request) -> Response {
    Response::ok()
}
EOF

git add api.rs
git commit -m "Add API handler"

echo "5. Capturing multiline note with metadata..."
code-notes capture api.rs \
  --author "Backend Team" \
  --collection api

echo ""
echo "=================================================="
echo "Example 4: Python with @meta:"
echo "=================================================="
echo ""

cat > validator.py << 'EOF'
# @note: Input validation is missing
# @meta: {"tags":["bug","security"],"severity":"high","priority":8}
def process_input(data):
    # No validation!
    return data.upper()

# @note: Add type hints here
# @meta: {"tags":["enhancement","python"],"priority":3}
def format_output(result):
    return str(result)
EOF

git add validator.py
git commit -m "Add validator"

echo "6. Capturing Python notes with inline metadata..."
code-notes capture validator.py \
  --author "Python Team" \
  --collection python-issues

echo ""
echo "=================================================="
echo "Viewing Captured Notes with Metadata"
echo "=================================================="
echo ""

# List all notes
echo "7. Listing all notes..."
code-notes list

echo ""
echo "=================================================="
echo "View Individual Note with Metadata"
echo "=================================================="
echo ""

# Get a note ID and view it
echo "8. Viewing detailed note with metadata..."
SECURITY_NOTE=$(code-notes list --collection security | grep "ID:" | head -1 | awk '{print $2}')
if [ ! -z "$SECURITY_NOTE" ]; then
    echo "Note details:"
    code-notes view "$SECURITY_NOTE"
fi

echo ""
echo "=================================================="
echo "Metadata Merging Rules"
echo "=================================================="
echo ""

cat << 'EOF'
How inline @meta: works:

1. CLI metadata (--metadata flag) acts as DEFAULT for all captured notes
2. Inline @meta: can override or add to CLI metadata per-note
3. If both CLI and inline have the same key, inline wins

Example:

  CLI: --metadata '{"tags":["default"],"priority":5}'

  Note 1: @meta: {"priority":9}
  Result: {"tags":["default"],"priority":9}

  Note 2: (no @meta:)
  Result: {"tags":["default"],"priority":5}

EOF

echo ""
echo "=================================================="
echo "@meta: Syntax Examples"
echo "=================================================="
echo ""

cat << 'EOF'
Valid @meta: syntax:

Rust/JavaScript/TypeScript:
  // @note: Fix this bug
  // @meta: {"tags":["bug"],"priority":8}

Python/Ruby/Shell:
  # @note: Refactor this function
  # @meta: {"tags":["refactor"],"effort":"medium"}

Complex metadata:
  // @note: Performance issue
  // @meta: {"tags":["performance"],"baseline_ms":150,"target_ms":50,"profiled":true}

With linked issues:
  // @note: Security vulnerability
  // @meta: {"tags":["security"],"severity":"critical","linked_issues":["CVE-2024-001","SEC-123"]}

Learning trails:
  // @note: Start here for authentication
  // @meta: {"trail_id":"auth-flow","step":1,"audience":"junior-devs"}

EOF

echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="
echo ""
echo "✓ Use @meta: on the line after @note: to specify metadata"
echo "✓ @meta: takes JSON format"
echo "✓ Inline metadata can be different for each note"
echo "✓ CLI --metadata provides defaults for all notes"
echo "✓ Inline @meta: overrides CLI metadata for same keys"
echo "✓ Works with all supported languages (Rust, Python, JS, TS, etc.)"
echo ""
echo "Demo directory: $DEMO_DIR"
echo "Notes storage: ~/.code-notes/"
echo ""
