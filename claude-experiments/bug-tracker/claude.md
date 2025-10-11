
## Bug Tracker

Use this tool to record bugs discovered during development. This helps track issues that need to be addressed later. Each bug gets a unique ID (goofy animal name like "curious-elephant") for easy reference and closing.

### Tool Definition

```json
{
  "name": "bug_tracker",
  "description": "Records bugs discovered during development to BUGS.md in the project root. Each bug gets a unique goofy animal name ID. Includes AI-powered quality validation.",
  "input_schema": {
    "type": "object",
    "properties": {
      "project": {
        "type": "string",
        "description": "Project root directory path"
      },
      "title": {
        "type": "string",
        "description": "Short bug title/summary"
      },
      "description": {
        "type": "string",
        "description": "Detailed description of the bug"
      },
      "file": {
        "type": "string",
        "description": "File path where bug was found"
      },
      "context": {
        "type": "string",
        "description": "Code context like function/class/module name where bug was found"
      },
      "severity": {
        "type": "string",
        "enum": ["low", "medium", "high", "critical"],
        "description": "Bug severity level"
      },
      "tags": {
        "type": "string",
        "description": "Comma-separated tags"
      },
      "repro": {
        "type": "string",
        "description": "Minimal reproducing case or steps to reproduce"
      },
      "code_snippet": {
        "type": "string",
        "description": "Code snippet demonstrating the bug"
      },
      "metadata": {
        "type": "string",
        "description": "Additional metadata as JSON string (e.g., version, platform)"
      }
    },
    "required": ["project", "title"]
  }
}
```

### Usage

Add a bug:
```bash
bug-tracker add --title <TITLE> [OPTIONS]
```

Close a bug:
```bash
bug-tracker close <BUG_ID>
```

List bugs:
```bash
bug-tracker list
```

### Examples

**Add a comprehensive bug report:**
```bash
bug-tracker add --title "Null pointer dereference" --description "Found potential null pointer access" --file "src/main.rs" --context "authenticate()" --severity high --tags "memory,safety" --repro "Call authenticate with null user_ptr" --code-snippet "if (!user_ptr) { /* missing check */ }"
```

**Close a bug by ID:**
```bash
bug-tracker close curious-elephant
```

**Enable AI quality validation:**
```bash
bug-tracker add --title "Bug title" --description "Bug details" --validate
```

The `--validate` flag triggers AI-powered quality checking to ensure bug reports contain sufficient information before recording.
