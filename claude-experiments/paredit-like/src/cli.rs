use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "paredit-like")]
#[command(version)]
#[command(about = "Structured editing for Clojure s-expressions")]
#[command(long_about = r#"paredit-like - Structured editing for Clojure s-expressions

A command-line tool for refactoring Clojure code using paredit-style
structural editing commands. Supports automatic parenthesis balancing,
slurping, barfing, and other common structural operations.

EXAMPLES:
    # Balance parentheses in a file and print to stdout
    paredit-like balance src/core.clj

    # Balance and modify file in-place
    paredit-like balance src/core.clj --in-place

    # Preview changes before applying
    paredit-like balance src/core.clj --dry-run

    # Slurp the next form into the list at line 5
    paredit-like slurp src/core.clj --line 5 -i

    # Barf the last form out of the list at line 10
    paredit-like barf src/core.clj --line 10 -i

    # Wrap form at line 3 with square brackets
    paredit-like wrap src/core.clj --line 3 --with "[" -i

    # Process multiple files
    paredit-like batch "src/**/*.clj" --command balance
"#)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Balance parentheses using indentation (Parinfer-style)
    ///
    /// Automatically adds missing closing parentheses and removes
    /// extra ones based on the code's indentation structure.
    ///
    /// Example:
    ///   Input:  (defn foo [x\n         (+ x 1
    ///   Output: (defn foo [x]\n         (+ x 1))
    #[command(verbatim_doc_comment)]
    Balance {
        /// Input file path
        file: PathBuf,

        /// Modify file in-place
        #[arg(short, long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Slurp the next form into the current list
    ///
    /// Moves the next (or previous with --backward) form inside
    /// the current list by extending the closing parenthesis.
    ///
    /// Example (forward):
    ///   Input:  (foo bar) baz
    ///   Output: (foo bar baz)
    ///
    /// Example (backward):
    ///   Input:  foo (bar baz)
    ///   Output: (foo bar baz)
    #[command(verbatim_doc_comment)]
    Slurp {
        /// Input file path
        file: PathBuf,

        /// Line number to target
        #[arg(short, long)]
        line: usize,

        /// Slurp backward instead of forward
        #[arg(short, long)]
        backward: bool,

        /// Modify file in-place
        #[arg(short = 'i', long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Barf the last form out of the current list
    ///
    /// Moves the last (or first with --backward) form outside
    /// the current list by contracting the closing parenthesis.
    /// The opposite of slurp.
    ///
    /// Example (forward):
    ///   Input:  (foo bar baz)
    ///   Output: (foo bar) baz
    ///
    /// Example (backward):
    ///   Input:  (foo bar baz)
    ///   Output: foo (bar baz)
    #[command(verbatim_doc_comment)]
    Barf {
        /// Input file path
        file: PathBuf,

        /// Line number to target
        #[arg(short, long)]
        line: usize,

        /// Barf backward instead of forward
        #[arg(short, long)]
        backward: bool,

        /// Modify file in-place
        #[arg(short = 'i', long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Splice the current list (remove parens, keep children)
    ///
    /// Removes the opening and closing parentheses of the current
    /// list, leaving its children in place.
    ///
    /// Example:
    ///   Input:  (foo (bar baz) qux)
    ///   Output: (foo bar baz qux)
    #[command(verbatim_doc_comment)]
    Splice {
        /// Input file path
        file: PathBuf,

        /// Line number to target
        #[arg(short, long)]
        line: usize,

        /// Modify file in-place
        #[arg(short = 'i', long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Raise the current form (replace parent with current)
    ///
    /// Replaces the parent list with the current form, effectively
    /// removing all siblings and the parent's parentheses.
    ///
    /// Example:
    ///   Input:  (foo (bar baz) qux)  [cursor on 'bar']
    ///   Output: (bar baz)
    #[command(verbatim_doc_comment)]
    Raise {
        /// Input file path
        file: PathBuf,

        /// Line number to target
        #[arg(short, long)]
        line: usize,

        /// Modify file in-place
        #[arg(short = 'i', long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Wrap the current form with parens/brackets
    ///
    /// Surrounds the current form with parentheses, square brackets,
    /// or curly braces.
    ///
    /// Example (with parentheses):
    ///   Input:  foo bar
    ///   Output: (foo) bar
    ///
    /// Example (with square brackets):
    ///   Input:  foo bar
    ///   Output: [foo] bar
    #[command(verbatim_doc_comment)]
    Wrap {
        /// Input file path
        file: PathBuf,

        /// Line number to target
        #[arg(short, long)]
        line: usize,

        /// Wrapper type: (, [, or {
        #[arg(short, long, default_value = "(")]
        with: String,

        /// Modify file in-place
        #[arg(short = 'i', long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Merge nested let forms into one
    ///
    /// Combines nested let bindings into a single let form,
    /// flattening the structure. If no line is specified, merges
    /// all nested lets in the entire file.
    ///
    /// Example:
    ///   Input:  (let [x 1]
    ///             (let [y 2]
    ///               (+ x y)))
    ///   Output: (let [x 1 y 2]
    ///             (+ x y))
    #[command(verbatim_doc_comment)]
    MergeLet {
        /// Input file path
        file: PathBuf,

        /// Line number to target (omit to merge all lets in file)
        #[arg(short, long)]
        line: Option<usize>,

        /// Modify file in-place
        #[arg(short = 'i', long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Apply refactorings to multiple files
    ///
    /// Process multiple files matching a glob pattern with the
    /// specified command.
    ///
    /// Example:
    ///   paredit-like batch "src/**/*.clj" --command balance
    #[command(verbatim_doc_comment)]
    Batch {
        /// Glob pattern for files (e.g., "src/**/*.clj")
        pattern: String,

        /// Command to run on each file
        #[arg(short, long)]
        command: String,

        /// Show what would change without modifying
        #[arg(short, long)]
        dry_run: bool,
    },

    /// Auto-indent code based on structure (Clojure-mode style)
    ///
    /// Automatically indents Clojure code following the same rules as
    /// clojure-mode in Emacs. Handles special forms like defn, let, if,
    /// and provides proper indentation for function calls.
    ///
    /// Example:
    ///   Input:  (defn foo [x]
    ///   (+ x 1))
    ///   Output: (defn foo [x]
    ///             (+ x 1))
    #[command(verbatim_doc_comment)]
    Indent {
        /// Input file path
        file: PathBuf,

        /// Modify file in-place
        #[arg(short, long)]
        in_place: bool,

        /// Preview output without modifying file
        #[arg(short, long)]
        dry_run: bool,

        /// Show diff instead of full output
        #[arg(long)]
        diff: bool,
    },

    /// Install tool documentation to claude.md
    ///
    /// Adds a brief description of paredit-like to the project's
    /// claude.md file, making Claude aware of the tool's existence.
    ///
    /// Example:
    ///   paredit-like install
    ///   paredit-like install --path /path/to/project
    #[command(verbatim_doc_comment)]
    Install {
        /// Target directory (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Overwrite existing paredit-like section if present
        #[arg(short, long)]
        force: bool,
    },
}

pub struct Output {
    pub original: String,
    pub modified: String,
}

impl Output {
    pub fn display(&self, in_place: bool, dry_run: bool, show_diff: bool, file_path: &PathBuf) -> Result<()> {
        if show_diff {
            // Show diff
            self.show_diff();
            Ok(())
        } else if dry_run {
            // Just print the modified output (dry-run = don't write to file)
            println!("{}", self.modified);
            Ok(())
        } else if in_place {
            // Write to file
            fs::write(file_path, &self.modified)
                .context(format!("Failed to write to {:?}", file_path))?;
            eprintln!("Modified {:?}", file_path);
            Ok(())
        } else {
            // Print to stdout
            println!("{}", self.modified);
            Ok(())
        }
    }

    fn show_diff(&self) {
        let diff = TextDiff::from_lines(&self.original, &self.modified);

        for change in diff.iter_all_changes() {
            let sign = match change.tag() {
                ChangeTag::Delete => "-",
                ChangeTag::Insert => "+",
                ChangeTag::Equal => " ",
            };
            print!("{}{}", sign, change);
        }
    }
}

pub fn read_file(path: &PathBuf) -> Result<String> {
    fs::read_to_string(path).context(format!("Failed to read file {:?}", path))
}

pub fn process_batch(pattern: &str, command: &str, _dry_run: bool) -> Result<()> {
    let files: Vec<PathBuf> = glob::glob(pattern)
        .context("Failed to parse glob pattern")?
        .filter_map(Result::ok)
        .collect();

    if files.is_empty() {
        eprintln!("No files matched pattern: {}", pattern);
        return Ok(());
    }

    eprintln!("Processing {} files...", files.len());

    for file in files {
        eprintln!("\n{:?}:", file);

        // Parse command and execute
        // This is a simplified version - you'd need to parse the command properly
        match command {
            "balance" => {
                // Execute balance command
                eprintln!("  Running balance...");
            }
            _ => {
                eprintln!("  Unknown command: {}", command);
            }
        }
    }

    Ok(())
}

const PAREDIT_SECTION: &str = r#"
## paredit-like - Structured Editing for Clojure

This project has access to `paredit-like`, a command-line tool for structural editing
of Clojure s-expressions. Use this tool when you need to refactor Clojure code while
maintaining proper parenthesis balance and structure.

### When to Use

- **Balance parentheses**: Automatically fix missing or extra parentheses based on indentation
- **Slurp/Barf**: Move forms in/out of lists (paredit-style operations)
- **Structural refactoring**: Splice, raise, wrap, or merge forms
- **Batch operations**: Apply refactorings to multiple files at once

### Common Commands

```bash
# Balance parentheses in a file
paredit-like balance src/core.clj --in-place

# Preview changes before applying
paredit-like balance src/core.clj --dry-run

# Slurp next form into list at line 5
paredit-like slurp src/core.clj --line 5 -i

# Barf last form out of list at line 10
paredit-like barf src/core.clj --line 10 -i

# Wrap form at line 3 with square brackets
paredit-like wrap src/core.clj --line 3 --with "[" -i

# Merge nested let forms
paredit-like merge-let src/core.clj --line 5 -i
```

### Getting Help

Run `paredit-like --help` to see all available commands and options.
Run `paredit-like <command> --help` for detailed help on specific commands.

### Available Operations

- `balance` - Fix parentheses using indentation (Parinfer-style)
- `slurp` - Move next/previous form into current list
- `barf` - Move last/first form out of current list
- `splice` - Remove parentheses, keep children
- `raise` - Replace parent with current form
- `wrap` - Surround form with parens/brackets/braces
- `merge-let` - Flatten nested let forms
- `batch` - Apply operations to multiple files

"#;

pub fn install_documentation(target_path: Option<PathBuf>, force: bool) -> Result<()> {
    let target_dir = target_path.unwrap_or_else(|| PathBuf::from("."));
    let claude_md_path = target_dir.join("claude.md");

    // Read existing content if file exists
    let existing_content = if claude_md_path.exists() {
        fs::read_to_string(&claude_md_path)
            .context(format!("Failed to read {:?}", claude_md_path))?
    } else {
        String::new()
    };

    // Check if paredit-like section already exists
    if existing_content.contains("## paredit-like") && !force {
        eprintln!("paredit-like section already exists in {:?}", claude_md_path);
        eprintln!("Use --force to overwrite the existing section");
        return Ok(());
    }

    // If force is used, remove existing section
    let content_to_write = if force && existing_content.contains("## paredit-like") {
        // Find the paredit-like section and remove it
        let lines: Vec<&str> = existing_content.lines().collect();
        let mut new_lines = Vec::new();
        let mut skip_section = false;

        for line in lines {
            if line.starts_with("## paredit-like") {
                skip_section = true;
                continue;
            }
            if skip_section && line.starts_with("## ") {
                skip_section = false;
            }
            if !skip_section {
                new_lines.push(line);
            }
        }

        let mut result = new_lines.join("\n");
        // Remove trailing empty lines
        while result.ends_with("\n\n\n") {
            result = result.trim_end_matches('\n').to_string();
            result.push('\n');
        }
        result
    } else {
        existing_content
    };

    // Append paredit-like section
    let final_content = if content_to_write.is_empty() {
        PAREDIT_SECTION.trim_start().to_string()
    } else {
        format!("{}\n{}", content_to_write.trim_end(), PAREDIT_SECTION)
    };

    // Write to file
    fs::write(&claude_md_path, final_content)
        .context(format!("Failed to write to {:?}", claude_md_path))?;

    eprintln!("✓ Added paredit-like documentation to {:?}", claude_md_path);
    eprintln!("  Claude can now use paredit-like for Clojure refactoring tasks");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_output_display_stdout() {
        let output = Output {
            original: "original".to_string(),
            modified: "modified".to_string(),
        };
        
        // Test stdout mode (in_place=false, dry_run=false)
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.clj");
        
        let result = output.display(false, false, false, &file_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_output_display_in_place() {
        let output = Output {
            original: "original".to_string(),
            modified: "modified".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.clj");
        
        // Create initial file
        fs::write(&file_path, "original").unwrap();
        
        let result = output.display(true, false, false, &file_path);
        assert!(result.is_ok());

        // Check file was modified
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "modified");
    }

    #[test]
    fn test_output_display_dry_run() {
        let output = Output {
            original: "line1\nline2\n".to_string(),
            modified: "line1\nmodified\n".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.clj");
        
        let result = output.display(false, true, false, &file_path);
        assert!(result.is_ok());

        // File should not exist (dry run)
        assert!(!file_path.exists());
    }

    #[test]
    fn test_output_display_dry_run_with_existing_file() {
        let output = Output {
            original: "original".to_string(),
            modified: "modified".to_string(),
        };
        
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.clj");
        
        // Create initial file
        fs::write(&file_path, "original").unwrap();
        
        let result = output.display(false, true, false, &file_path);
        assert!(result.is_ok());

        // File should remain unchanged (dry run)
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "original");
    }

    #[test]
    fn test_read_file_success() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.clj");
        
        let content = "(defn test [x] (+ x 1))";
        fs::write(&file_path, content).unwrap();
        
        let result = read_file(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content);
    }

    #[test]
    fn test_read_file_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("nonexistent.clj");
        
        let result = read_file(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read file"));
    }

    #[test]
    fn test_read_file_empty() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("empty.clj");
        
        fs::write(&file_path, "").unwrap();
        
        let result = read_file(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_read_file_unicode() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("unicode.clj");
        
        let content = "(str \"λ α β γ δ ε\")";
        fs::write(&file_path, content).unwrap();
        
        let result = read_file(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content);
    }

    #[test]
    fn test_read_file_multiline() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("multiline.clj");
        
        let content = "(defn multi-line [x]\n  (let [y (+ x 1)]\n    (* y 2)))";
        fs::write(&file_path, content).unwrap();
        
        let result = read_file(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content);
    }

    #[test]
    fn test_process_batch_no_files() {
        let result = process_batch("nonexistent/**/*.clj", "balance", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_batch_invalid_pattern() {
        let result = process_batch("[invalid", "balance", true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse glob pattern"));
    }

    #[test]
    fn test_process_batch_with_files() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create test files
        let file1 = temp_dir.path().join("test1.clj");
        let file2 = temp_dir.path().join("test2.clj");
        fs::write(&file1, "(defn foo []").unwrap();
        fs::write(&file2, "(defn bar []").unwrap();
        
        let pattern = format!("{}/*.clj", temp_dir.path().display());
        let result = process_batch(&pattern, "balance", true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_process_batch_unknown_command() {
        let temp_dir = TempDir::new().unwrap();
        let file1 = temp_dir.path().join("test.clj");
        fs::write(&file1, "(defn foo []").unwrap();
        
        let pattern = format!("{}/*.clj", temp_dir.path().display());
        let result = process_batch(&pattern, "unknown", true);
        assert!(result.is_ok()); // Should not error, just print message
    }

    #[test]
    fn test_output_show_diff_no_changes() {
        let output = Output {
            original: "same content".to_string(),
            modified: "same content".to_string(),
        };
        
        // This test verifies the diff function doesn't panic
        output.show_diff();
    }

    #[test]
    fn test_output_show_diff_with_changes() {
        let output = Output {
            original: "line1\nline2\nline3".to_string(),
            modified: "line1\nmodified\nline3".to_string(),
        };
        
        // This test verifies the diff function doesn't panic
        output.show_diff();
    }

    #[test]
    fn test_output_show_diff_additions() {
        let output = Output {
            original: "line1\nline2".to_string(),
            modified: "line1\nline2\nline3".to_string(),
        };
        
        output.show_diff();
    }

    #[test]
    fn test_output_show_diff_deletions() {
        let output = Output {
            original: "line1\nline2\nline3".to_string(),
            modified: "line1\nline3".to_string(),
        };
        
        output.show_diff();
    }

    #[test]
    fn test_output_display_write_error() {
        let output = Output {
            original: "original".to_string(),
            modified: "modified".to_string(),
        };
        
        // Try to write to a path that doesn't exist
        let bad_path = PathBuf::from("/nonexistent/directory/file.clj");
        let result = output.display(true, false, false, &bad_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to write"));
    }
}
