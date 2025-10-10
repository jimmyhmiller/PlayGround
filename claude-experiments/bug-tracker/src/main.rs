use chrono::Local;
use clap::{Parser, Subcommand};
use goofy_animals::generate_name;
use rand::rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

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

    /// Install the tool definition in claude.md
    Install {
        /// Path to the claude.md file (defaults to .claude/claude.md in current directory)
        #[arg(short, long)]
        claude_md: Option<PathBuf>,
    },
}

#[derive(Serialize, Deserialize)]
struct BugEntry {
    id: String,
    timestamp: String,
    title: String,
    description: Option<String>,
    file: Option<String>,
    context: Option<String>,
    severity: String,
    tags: Vec<String>,
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
        } => {
            if let Err(e) = add_bug(project, title, description, file, context, severity, tags) {
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
        Commands::Install { claude_md } => {
            if let Err(e) = install_to_claude_md(claude_md) {
                eprintln!("Error installing to claude.md: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn add_bug(
    project: Option<PathBuf>,
    title: String,
    description: Option<String>,
    file: Option<String>,
    context: Option<String>,
    severity: String,
    tags: Option<String>,
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

    let bug = BugEntry {
        id: bug_id.clone(),
        timestamp,
        title: title.clone(),
        description: description.clone(),
        file: file.clone(),
        context: context.clone(),
        severity: severity.clone(),
        tags: tag_list.clone(),
    };

    // Create or append to BUGS.md
    let mut output = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&bugs_file)?;

    // Check if file is empty (just created)
    let metadata = fs::metadata(&bugs_file)?;
    if metadata.len() == 0 {
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
    writeln!(output, "**Timestamp:** {}", bug.timestamp)?;
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
    let bug_pattern = format!(r"(?s)## .*?\[{}\].*?---\s*\n", regex::escape(&bug_id));
    let re = Regex::new(&bug_pattern)?;

    if !re.is_match(&content) {
        return Err(format!("Bug with ID '{}' not found", bug_id).into());
    }

    let new_content = re.replace(&content, "");

    // Write back the modified content
    fs::write(&bugs_file, new_content.as_ref())?;

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

Use this tool to record bugs discovered during development. This helps track issues that need to be addressed later.

### Tool Definition

```json
{
  "name": "bug_tracker",
  "description": "Records bugs discovered during development to BUGS.md in the project root. Each bug gets a unique goofy animal name ID.",
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
      }
    },
    "required": ["project", "title"]
  }
}
```

### Usage

Add a bug:
```bash
bug-tracker add --project <PATH> --title <TITLE> [OPTIONS]
```

Close a bug:
```bash
bug-tracker close --project <PATH> <BUG_ID>
```

List bugs:
```bash
bug-tracker list --project <PATH>
```

### Example

```bash
bug-tracker add --project /path/to/project --title "Null pointer dereference" --description "Found potential null pointer access" --file "src/main.rs" --context "authenticate()" --severity high --tags "memory,safety"
```
"#;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&claude_md_path)?;

    writeln!(file, "{}", tool_definition)?;

    println!("✓ Tool definition added to {}", claude_md_path.display());
    println!("\nNote: You may need to restart your Claude Code session for changes to take effect.");
    Ok(())
}
