#!/bin/bash
# Install git_guard hooks for Claude Code into current project
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-.}"
HOOKS_DIR="$PROJECT_DIR/.claude/hooks"
SETTINGS_FILE="$PROJECT_DIR/.claude/settings.json"

echo "Installing git_guard to: $PROJECT_DIR"

# Create hooks directory
mkdir -p "$HOOKS_DIR"

# Copy files
cp "$SCRIPT_DIR/git_guard.py" "$HOOKS_DIR/"
chmod +x "$HOOKS_DIR/git_guard.py"

# Create config if it doesn't exist
if [ ! -f "$HOOKS_DIR/git_guard.json" ]; then
    cat > "$HOOKS_DIR/git_guard.json" << 'EOF'
{
  "block_stash": true,
  "protected_branches": ["main", "master"]
}
EOF
fi

# Create or update settings.json
HOOK_CONFIG='{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 $CLAUDE_PROJECT_DIR/.claude/hooks/git_guard.py"
          }
        ]
      }
    ]
  }
}'

if [ -f "$SETTINGS_FILE" ]; then
    # Merge with existing settings using jq if available
    if command -v jq &> /dev/null; then
        jq -s '.[0] * .[1]' "$SETTINGS_FILE" <(echo "$HOOK_CONFIG") > "$SETTINGS_FILE.tmp"
        mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"
    else
        echo "Warning: jq not found. Please manually merge hooks into $SETTINGS_FILE"
        echo "$HOOK_CONFIG"
    fi
else
    mkdir -p "$(dirname "$SETTINGS_FILE")"
    echo "$HOOK_CONFIG" > "$SETTINGS_FILE"
fi

echo "Done! Restart Claude Code for hooks to take effect."
echo "Edit $HOOKS_DIR/git_guard.json to configure."
