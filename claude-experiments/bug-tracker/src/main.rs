use chrono::Local;
use clap::{Parser, Subcommand};
use goofy_animals::generate_name;
use rand::rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser)]
#[command(name = "bug-tracker")]
#[command(about = "A CLI tool for LLM agents to track bugs during development", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a new bug to BUGS.md
    Add {
        /// Project root directory (where BUGS.md will be created/updated). Defaults to searching current directory and parents.
        #[arg(short, long)]
        project: Option<PathBuf>,

        /// Bug title/summary
        #[arg(short, long)]
        title: String,

        /// Detailed description of the bug
        #[arg(short, long)]
        description: Option<String>,

        /// File path where the bug was found
        #[arg(short, long)]
        file: Option<String>,

        /// Code context or function name where the bug was found
        #[arg(short, long)]
        context: Option<String>,

        /// Severity level (low, medium, high, critical)
        #[arg(short, long, default_value = "medium")]
        severity: String,

        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,

        /// Minimal reproducing case
        #[arg(short = 'r', long)]
        repro: Option<String>,

        /// Code snippet demonstrating the bug
        #[arg(long)]
        code_snippet: Option<String>,

        /// Additional metadata in JSON format (e.g., '{"version":"1.2.3","platform":"linux"}')
        #[arg(long)]
        metadata: Option<String>,

        /// Enable AI quality check validation
        #[arg(long)]
        validate: bool,
    },

    /// Close a bug by its ID
    Close {
        /// Project root directory (where BUGS.md is located). Defaults to searching current directory and parents.
        #[arg(short, long)]
        project: Option<PathBuf>,

        /// Bug ID to close (e.g., "curious-elephant")
        bug_id: String,
    },

    /// List all bugs
    List {
        /// Project root directory (where BUGS.md is located). Defaults to searching current directory and parents.
        #[arg(short, long)]
        project: Option<PathBuf>,
    },

    /// View a specific bug by its ID
    View {
        /// Project root directory (where BUGS.md is located). Defaults to searching current directory and parents.
        #[arg(short, long)]
        project: Option<PathBuf>,

        /// Bug ID to view (e.g., "curious-elephant")
        bug_id: String,
    },

    /// Install the tool definition in claude.md
    Install {
        /// Path to the claude.md file (defaults to .claude/claude.md in current directory)
        #[arg(short, long)]
        claude_md: Option<PathBuf>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
struct BugEntry {
    id: String,
    timestamp: String,
    title: String,
    description: Option<String>,
    file: Option<String>,
    context: Option<String>,
    severity: String,
    tags: Vec<String>,
    repro: Option<String>,
    code_snippet: Option<String>,
    metadata: HashMap<String, Value>,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Add {
            project,
            title,
            description,
            file,
            context,
            severity,
            tags,
            repro,
            code_snippet,
            metadata,
            validate,
        } => {
            if let Err(e) = add_bug(
                project,
                title,
                description,
                file,
                context,
                severity,
                tags,
                repro,
                code_snippet,
                metadata,
                validate,
            ) {
                eprintln!("Error adding bug: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Close { project, bug_id } => {
            if let Err(e) = close_bug(project, bug_id) {
                eprintln!("Error closing bug: {}", e);
                std::process::exit(1);
            }
        }
        Commands::List { project } => {
            if let Err(e) = list_bugs(project) {
                eprintln!("Error listing bugs: {}", e);
                std::process::exit(1);
            }
        }
        Commands::View { project, bug_id } => {
            if let Err(e) = view_bug(project, bug_id) {
                eprintln!("Error viewing bug: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Install { claude_md } => {
            if let Err(e) = install_to_claude_md(claude_md) {
                eprintln!("Error installing to claude.md: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn validate_bug_report(bug: &BugEntry) -> Result<bool, Box<dyn std::error::Error>> {
    let bug_json = serde_json::to_string_pretty(&bug)?;

    let system_prompt = r#"You are a Bug Report Quality Analyst. Your job is to review bug reports and ensure they contain sufficient information to be actionable.

A good bug report should have:
1. A clear, descriptive title
2. A detailed description of the issue
3. Context about where the bug occurs (file, function, etc.)
4. Appropriate severity level
5. If applicable: reproduction steps, code snippets, or relevant metadata

Respond with a JSON object in this format:
{
  "approved": true/false,
  "feedback": "Detailed feedback on what's missing or why it's approved"
}

If the bug report is minimal but acceptable (e.g., for a simple issue), you can approve it.
If it lacks critical information that would make it hard to fix, reject it with specific feedback."#;

    let user_prompt = format!("Please review this bug report:\n\n{}", bug_json);

    // Call claude CLI - use full path from HOME
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/jimmyhmiller".to_string());
    let claude_path = format!("{}/.claude/local/claude", home);

    let output = Command::new(&claude_path)
        .arg("--print")
        .arg("--append-system-prompt")
        .arg(system_prompt)
        .arg(user_prompt)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("Claude CLI failed with status: {:?}", output.status);
        eprintln!("STDOUT: {}", stdout);
        eprintln!("STDERR: {}", stderr);
        return Err("Failed to run claude CLI for validation".into());
    }

    let response = String::from_utf8_lossy(&output.stdout);

    // Try to extract JSON from the response
    let json_start = response.find('{');
    let json_end = response.rfind('}');

    if let (Some(start), Some(end)) = (json_start, json_end) {
        let json_str = &response[start..=end];
        let validation: Value = serde_json::from_str(json_str)?;

        let approved = validation["approved"].as_bool().unwrap_or(false);
        let feedback = validation["feedback"].as_str().unwrap_or("No feedback provided");

        println!("\n=== Bug Report Quality Check ===");
        println!("{}", feedback);
        println!("================================\n");

        if !approved {
            println!("Bug report needs improvement. Please provide more details.");
            return Ok(false);
        }
    } else {
        eprintln!("Warning: Could not parse validation response, proceeding anyway");
    }

    Ok(true)
}

fn add_bug(
    project: Option<PathBuf>,
    title: String,
    description: Option<String>,
    file: Option<String>,
    context: Option<String>,
    severity: String,
    tags: Option<String>,
    repro: Option<String>,
    code_snippet: Option<String>,
    metadata: Option<String>,
    validate: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let project_root = if let Some(p) = project {
        p
    } else {
        // For add, use current directory if no BUGS.md found
        find_bugs_file(None).unwrap_or_else(|_| std::env::current_dir().unwrap())
    };

    let bugs_file = project_root.join("BUGS.md");
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let mut rng = rng();
    let bug_id = generate_name(&mut rng);

    let tag_list: Vec<String> = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let metadata_map: HashMap<String, Value> = if let Some(meta_str) = metadata {
        serde_json::from_str(&meta_str).unwrap_or_default()
    } else {
        HashMap::new()
    };

    let bug = BugEntry {
        id: bug_id.clone(),
        timestamp: timestamp.clone(),
        title: title.clone(),
        description: description.clone(),
        file: file.clone(),
        context: context.clone(),
        severity: severity.clone(),
        tags: tag_list.clone(),
        repro: repro.clone(),
        code_snippet: code_snippet.clone(),
        metadata: metadata_map.clone(),
    };

    // Validate the bug report if requested
    if validate {
        match validate_bug_report(&bug) {
            Ok(approved) => {
                if !approved {
                    return Err("Bug report did not pass quality check. Please revise and try again.".into());
                }
            }
            Err(e) => {
                eprintln!("Warning: Validation failed: {}. Proceeding anyway.", e);
            }
        }
    }

    // Create or append to BUGS.md
    let mut output = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&bugs_file)?;

    // Check if file is empty (just created)
    let file_metadata = fs::metadata(&bugs_file)?;
    if file_metadata.len() == 0 {
        writeln!(output, "<!-- ")?;
        writeln!(output, "═══════════════════════════════════════════════════════════════════════")?;
        writeln!(output, "⚠️  WARNING: DO NOT EDIT THIS FILE MANUALLY!")?;
        writeln!(output, "═══════════════════════════════════════════════════════════════════════")?;
        writeln!(output)?;
        writeln!(output, "This file is managed by bug-tracker CLI tool.")?;
        writeln!(output, "Manual edits may be overwritten or cause parsing errors.")?;
        writeln!(output)?;
        writeln!(output, "To add, close, or list bugs, use:")?;
        writeln!(output, "    bug-tracker --help")?;
        writeln!(output)?;
        writeln!(output, "═══════════════════════════════════════════════════════════════════════")?;
        writeln!(output, "-->")?;
        writeln!(output)?;
        writeln!(output, "# Bugs")?;
        writeln!(output)?;
        writeln!(
            output,
            "This file tracks bugs discovered during development.\n"
        )?;
    }

    // Format the bug entry
    writeln!(output, "## {} [{}]", title, bug_id)?;
    writeln!(output)?;
    writeln!(output, "**ID:** {}", bug_id)?;
    writeln!(output, "**Timestamp:** {}", timestamp)?;
    writeln!(output, "**Severity:** {}", severity)?;

    if let Some(ref f) = file {
        write!(output, "**Location:** {}", f)?;
        if let Some(ref ctx) = context {
            writeln!(output, " ({})", ctx)?;
        } else {
            writeln!(output)?;
        }
    } else if let Some(ref ctx) = context {
        writeln!(output, "**Context:** {}", ctx)?;
    }

    if !tag_list.is_empty() {
        writeln!(output, "**Tags:** {}", tag_list.join(", "))?;
    }

    writeln!(output)?;

    if let Some(ref desc) = description {
        writeln!(output, "### Description")?;
        writeln!(output)?;
        writeln!(output, "{}", desc)?;
        writeln!(output)?;
    }

    if let Some(ref r) = repro {
        writeln!(output, "### Minimal Reproducing Case")?;
        writeln!(output)?;
        writeln!(output, "{}", r)?;
        writeln!(output)?;
    }

    if let Some(ref snippet) = code_snippet {
        writeln!(output, "### Code Snippet")?;
        writeln!(output)?;
        writeln!(output, "```")?;
        writeln!(output, "{}", snippet)?;
        writeln!(output, "```")?;
        writeln!(output)?;
    }

    if !metadata_map.is_empty() {
        writeln!(output, "### Metadata")?;
        writeln!(output)?;
        for (key, value) in metadata_map.iter() {
            writeln!(output, "- **{}:** {}", key, value)?;
        }
        writeln!(output)?;
    }

    writeln!(output, "---")?;
    writeln!(output)?;

    println!("✓ Bug added with ID: {}", bug_id);
    println!("  File: {}", bugs_file.display());
    Ok(())
}

fn close_bug(project: Option<PathBuf>, bug_id: String) -> Result<(), Box<dyn std::error::Error>> {
    let project_root = if let Some(p) = project {
        p
    } else {
        find_bugs_file(None)?
    };

    let bugs_file = project_root.join("BUGS.md");

    if !bugs_file.exists() {
        return Err("BUGS.md not found in project directory".into());
    }

    let content = fs::read_to_string(&bugs_file)?;

    // Pattern to match a bug section with the given ID
    // We'll match line by line to avoid crossing into other bug sections
    let header_pattern = format!(r"(?m)^## .*?\[{}\]", regex::escape(&bug_id));
    let header_re = Regex::new(&header_pattern)?;

    // Find the bug header
    let header_match = header_re.find(&content)
        .ok_or_else(|| format!("Bug with ID '{}' not found", bug_id))?;

    let start = header_match.start();

    // Find the end by looking for the "---" separator after the header
    // But stop if we encounter another "## " header first
    let after_header = &content[header_match.end()..];
    let separator_pattern = Regex::new(r"(?m)^---\s*\n")?;
    let next_header_pattern = Regex::new(r"(?m)^## ")?;

    let separator_pos = separator_pattern.find(after_header)
        .ok_or_else(|| format!("Malformed bug entry for ID '{}'", bug_id))?;

    // Check if there's another header before the separator (would indicate corruption)
    if let Some(next_header) = next_header_pattern.find(after_header) {
        if next_header.start() < separator_pos.start() {
            return Err(format!("Malformed bug entry for ID '{}': found another header before separator", bug_id).into());
        }
    }

    // The end position is after the separator
    let end = header_match.end() + separator_pos.end();

    // Remove the bug section
    let mut new_content = String::new();
    new_content.push_str(&content[..start]);
    new_content.push_str(&content[end..]);

    // Write back the modified content
    fs::write(&bugs_file, &new_content)?;

    println!("✓ Bug '{}' closed and removed from {}", bug_id, bugs_file.display());
    Ok(())
}

fn find_bugs_file(start_dir: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut current = if let Some(dir) = start_dir {
        dir.canonicalize()?
    } else {
        std::env::current_dir()?
    };

    loop {
        let bugs_file = current.join("BUGS.md");
        if bugs_file.exists() {
            return Ok(current);
        }

        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            return Err("Could not find BUGS.md in current directory or any parent directory".into());
        }
    }
}

fn list_bugs(project: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let project_root = if let Some(p) = project {
        p
    } else {
        find_bugs_file(None)?
    };

    let bugs_file = project_root.join("BUGS.md");

    if !bugs_file.exists() {
        println!("No bugs found (BUGS.md doesn't exist)");
        return Ok(());
    }

    let file = fs::File::open(&bugs_file)?;
    let reader = BufReader::new(file);

    // Regex to extract bug info from headers like: ## Title [bug-id]
    let header_re = Regex::new(r"^## (.+?) \[(.+?)\]$")?;
    let severity_re = Regex::new(r"^\*\*Severity:\*\* (.+)$")?;

    let mut found_bugs = false;
    let mut current_title: Option<String> = None;
    let mut current_id: Option<String> = None;

    for line in reader.lines() {
        let line = line?;

        if let Some(caps) = header_re.captures(&line) {
            let title = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let id = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            current_title = Some(title.to_string());
            current_id = Some(id.to_string());
        } else if let Some(caps) = severity_re.captures(&line) {
            if let (Some(title), Some(id)) = (current_title.take(), current_id.take()) {
                let severity = caps.get(1).map(|m| m.as_str()).unwrap_or("unknown");
                println!("[{}] {} (severity: {})", id, title, severity);
                found_bugs = true;
            }
        }
    }

    if !found_bugs {
        println!("No bugs found");
    }

    Ok(())
}

fn view_bug(project: Option<PathBuf>, bug_id: String) -> Result<(), Box<dyn std::error::Error>> {
    let project_root = if let Some(p) = project {
        p
    } else {
        find_bugs_file(None)?
    };

    let bugs_file = project_root.join("BUGS.md");

    if !bugs_file.exists() {
        return Err("BUGS.md not found in project directory".into());
    }

    let content = fs::read_to_string(&bugs_file)?;

    // Pattern to match a bug section with the given ID
    let bug_pattern = format!(r"(?s)(## .*?\[{}\].*?)(---\s*\n)", regex::escape(&bug_id));
    let re = Regex::new(&bug_pattern)?;

    if let Some(caps) = re.captures(&content) {
        let bug_content = caps.get(1).map(|m| m.as_str()).unwrap_or("");

        println!("\n{}", bug_content.trim());
        println!("---\n");

        Ok(())
    } else {
        Err(format!("Bug with ID '{}' not found", bug_id).into())
    }
}

fn install_to_claude_md(claude_md: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    // Determine the claude.md path
    let claude_md_path = if let Some(path) = claude_md {
        path
    } else {
        // Try to find claude.md in common locations
        let dotclause_path = PathBuf::from(".claude/claude.md");
        let root_path = PathBuf::from("claude.md");

        if dotclause_path.exists() {
            dotclause_path
        } else if root_path.exists() {
            root_path
        } else {
            return Err("Could not find claude.md in .claude/claude.md or ./claude.md. Please specify path with --claude-md".into());
        }
    };

    let tool_definition = r#"
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

View a bug:
```bash
bug-tracker view <BUG_ID>
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

**View a bug by ID:**
```bash
bug-tracker view curious-elephant
```

**Enable AI quality validation:**
```bash
bug-tracker add --title "Bug title" --description "Bug details" --validate
```

The `--validate` flag triggers AI-powered quality checking to ensure bug reports contain sufficient information before recording.
"#;

    // Read existing content if file exists
    let existing_content = if claude_md_path.exists() {
        fs::read_to_string(&claude_md_path)?
    } else {
        String::new()
    };

    // Check if Bug Tracker section exists
    let bug_tracker_start = "## Bug Tracker";

    let new_content = if existing_content.contains(bug_tracker_start) {
        // Find the start of the Bug Tracker section
        let start_pos = existing_content.find(bug_tracker_start).unwrap();

        // Find the end - either the next "## " heading or end of file
        let after_start = &existing_content[start_pos..];
        let next_section_offset = after_start[bug_tracker_start.len()..]
            .find("\n## ")
            .map(|pos| pos + bug_tracker_start.len());

        let end_pos = if let Some(offset) = next_section_offset {
            start_pos + offset
        } else {
            existing_content.len()
        };

        // Build new content: before + new definition + after
        let mut content = String::new();
        content.push_str(&existing_content[..start_pos]);
        content.push_str(tool_definition.trim());
        if end_pos < existing_content.len() {
            content.push('\n');
            content.push_str(&existing_content[end_pos..]);
        } else {
            content.push('\n');
        }

        println!("✓ Updated existing Bug Tracker definition in {}", claude_md_path.display());
        content
    } else {
        // Append new section
        let mut content = existing_content;
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(tool_definition);
        println!("✓ Added Bug Tracker definition to {}", claude_md_path.display());
        content
    };

    // Write the updated content
    fs::write(&claude_md_path, new_content)?;

    println!("\nNote: You may need to restart your Claude Code session for changes to take effect.");
    Ok(())
}
