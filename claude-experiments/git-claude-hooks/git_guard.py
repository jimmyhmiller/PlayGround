#!/usr/bin/env python3
"""
Git Safety Guard for Claude Code
Blocks dangerous git operations on protected branches.
"""
import json
import sys
import re
import subprocess
import os


def get_current_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None


def load_config():
    """Load configuration from git_guard.json."""
    config_path = os.path.join(
        os.environ.get("CLAUDE_PROJECT_DIR", "."),
        ".claude", "hooks", "git_guard.json"
    )
    defaults = {
        "block_stash": True,
        "protected_branches": ["main", "master"]
    }
    try:
        with open(config_path) as f:
            config = json.load(f)
            return {**defaults, **config}
    except:
        return defaults


def ask_user(reason, command):
    """Return JSON to ask user for permission."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason
        }
    }


def main():
    # Read hook input
    try:
        input_data = json.load(sys.stdin)
    except:
        sys.exit(0)

    if input_data.get("tool_name") != "Bash":
        sys.exit(0)

    command = input_data.get("tool_input", {}).get("command", "")
    config = load_config()

    # Check: git stash (if enabled)
    if config["block_stash"] and re.search(r"\bgit\s+stash\b", command):
        print(json.dumps(ask_user(
            "git stash is blocked by git_guard. Allow this command?",
            command
        )))
        sys.exit(0)

    # Check: git commit/push on protected branches
    if re.search(r"\bgit\s+(commit|push)\b", command):
        branch = get_current_branch()
        if branch in config["protected_branches"]:
            print(json.dumps(ask_user(
                f"git commit/push on protected branch '{branch}' is blocked. Allow?",
                command
            )))
            sys.exit(0)

    # Allow all other commands
    sys.exit(0)


if __name__ == "__main__":
    main()
