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


def is_git_ref(name):
    """Check if name is a valid git ref (branch, tag, commit)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", name],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except:
        return False


def is_tracked_file(path):
    """Check if path is a tracked file in the repo."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except:
        return False


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

    # Check: git checkout <something> - determine if it's a file or branch
    checkout_match = re.search(r"\bgit\s+checkout\s+(.+)", command)
    if checkout_match:
        args = checkout_match.group(1).strip()
        # Skip if it's a branch creation operation
        if re.match(r"^(-b|-B|--orphan)\s", args):
            sys.exit(0)
        # Handle explicit file checkout: git checkout -- <file>
        if "--" in args:
            after_dashdash = args.split("--", 1)[1].strip()
            if after_dashdash:
                print(json.dumps(ask_user(
                    f"git checkout -- discards uncommitted changes to '{after_dashdash}'. Allow?",
                    command
                )))
                sys.exit(0)
        # Parse the arguments to find potential file paths
        parts = args.split()
        for part in parts:
            # Skip flags
            if part.startswith("-"):
                continue
            # Check if this is a tracked file (not a branch/ref)
            if is_tracked_file(part) and not is_git_ref(part):
                print(json.dumps(ask_user(
                    f"git checkout of file '{part}' will discard uncommitted changes. Allow?",
                    command
                )))
                sys.exit(0)
            # If it's both a file AND a ref, still warn (ambiguous)
            if is_tracked_file(part) and is_git_ref(part):
                print(json.dumps(ask_user(
                    f"'{part}' is both a file and a branch/ref. Allow?",
                    command
                )))
                sys.exit(0)

    # Allow all other commands
    sys.exit(0)


if __name__ == "__main__":
    main()
