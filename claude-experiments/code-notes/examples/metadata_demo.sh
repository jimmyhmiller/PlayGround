#!/bin/bash

# Metadata Feature Demonstration
# This script demonstrates how to use metadata with code-notes add and capture commands

set -e

echo "=================================================="
echo "Code Notes - Metadata Feature Demo"
echo "=================================================="
echo ""

# Create a temporary demo directory
DEMO_DIR=$(mktemp -d -t code-notes-metadata-demo)
cd "$DEMO_DIR"

echo "Demo directory: $DEMO_DIR"
echo ""

# Initialize git repo
echo "1. Initializing git repository..."
git init
git config user.name "Demo User"
git config user.email "demo@example.com"

# Create a sample file
echo "2. Creating sample code file..."
cat > auth.rs << 'EOF'
pub fn authenticate(username: &str, password: &str) -> bool {
    // TODO: Implement actual authentication
    username == "admin" && password == "secret"
}

pub fn authorize(user: &str, resource: &str) -> bool {
    // Check if user has access to resource
    user == "admin"
}
EOF

git add auth.rs
git commit -m "Initial commit"

# Initialize code-notes
echo "3. Initializing code-notes..."
code-notes init

echo ""
echo "=================================================="
echo "Example 1: Adding a note with metadata using 'add'"
echo "=================================================="
echo ""

# Add a note with metadata
echo "4. Adding note with tags and priority metadata..."
code-notes add \
  --file auth.rs \
  --line 1 \
  --column 0 \
  --content "Security issue: This authentication is not secure. Uses plaintext comparison." \
  --author "Security Team" \
  --collection security \
  --metadata '{"tags":["bug","security","high-priority"],"priority":9,"severity":"critical","linked_issues":["SEC-123"]}'

echo ""
echo "5. Adding another note with different metadata..."
code-notes add \
  --file auth.rs \
  --line 6 \
  --column 0 \
  --content "This needs role-based access control (RBAC)" \
  --author "Architecture Team" \
  --collection architecture \
  --metadata '{"tags":["enhancement","rbac"],"priority":7,"audience":"senior-devs","trail_id":"auth-modernization"}'

echo ""
echo "=================================================="
echo "Example 2: Using capture with metadata"
echo "=================================================="
echo ""

# Create a file with @note: markers
echo "6. Creating a file with @note: markers..."
cat > config.rs << 'EOF'
// @note: Configuration loader for the application
pub struct Config {
    pub database_url: String,
    pub api_key: String,
}

// @note: Load configuration from environment variables
// This should be replaced with a proper config file system
impl Config {
    pub fn from_env() -> Result<Self, String> {
        Ok(Config {
            database_url: std::env::var("DATABASE_URL").unwrap_or_default(),
            api_key: std::env::var("API_KEY").unwrap_or_default(),
        })
    }
}
EOF

git add config.rs
git commit -m "Add config module"

echo "7. Capturing all notes with onboarding metadata..."
code-notes capture config.rs \
  --author "Onboarding Team" \
  --collection onboarding \
  --metadata '{"tags":["onboarding","config"],"audience":"junior-devs","priority":5,"trail_id":"getting-started"}'

echo ""
echo "=================================================="
echo "Viewing Notes with Metadata"
echo "=================================================="
echo ""

# List all notes
echo "8. Listing all notes..."
code-notes list

echo ""
echo "=================================================="
echo "View Individual Notes (with metadata)"
echo "=================================================="
echo ""

# Get note IDs and view them
echo "9. Viewing detailed note information..."
SECURITY_NOTE=$(code-notes list --collection security | grep "ID:" | head -1 | awk '{print $2}')
if [ ! -z "$SECURITY_NOTE" ]; then
    echo "Security note details:"
    code-notes view "$SECURITY_NOTE"
fi

echo ""
echo "=================================================="
echo "Metadata Use Cases"
echo "=================================================="
echo ""

cat << 'EOF'
Common Metadata Patterns:

1. Bug Tracking:
   --metadata '{"tags":["bug"],"priority":8,"severity":"high","linked_issues":["BUG-456"]}'

2. Learning Trails:
   --metadata '{"trail_id":"auth-flow","step":1,"audience":"junior-devs","difficulty":"intermediate"}'

3. Code Review:
   --metadata '{"tags":["code-review"],"reviewer":"alice","date":"2024-01-15","status":"needs-fixing"}'

4. Technical Debt:
   --metadata '{"tags":["tech-debt"],"priority":6,"effort":"medium","impact":"high"}'

5. Documentation:
   --metadata '{"tags":["docs"],"audience":"all-devs","category":"api","related_docs":["api-guide.md"]}'

6. Performance:
   --metadata '{"tags":["performance"],"baseline_ms":150,"target_ms":50,"profiled":true}'

EOF

echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="
echo ""
echo "✓ Metadata can be added to notes via both 'add' and 'capture' commands"
echo "✓ Use --metadata flag with JSON string"
echo "✓ Metadata is displayed in 'view' command"
echo "✓ Metadata is preserved across migrations"
echo "✓ Use metadata for: tags, priorities, trails, audiences, issue linking, etc."
echo ""
echo "Demo directory preserved at: $DEMO_DIR"
echo "You can explore the notes storage at: ~/.code-notes/"
echo ""
