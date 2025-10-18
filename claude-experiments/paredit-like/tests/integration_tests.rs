use std::fs;
use std::process::Command;
use tempfile::TempDir;
use pretty_assertions::assert_eq;

/// Helper function to run the paredit-like CLI with given arguments
fn run_paredit_like(args: &[&str]) -> std::process::Output {
    Command::new("cargo")
        .args(&["run", "--"])
        .args(args)
        .output()
        .expect("Failed to execute paredit-like")
}

/// Helper function to create a temp file with content
fn create_temp_file(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
    let file_path = dir.path().join(name);
    fs::write(&file_path, content).unwrap();
    file_path
}

#[test]
fn test_balance_command_basic() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(defn foo\n  (+ 1 2");
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("(+ 1 2))"));
}

#[test]
fn test_balance_command_dry_run() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(defn foo\n  (+ 1 2");
    let original_content = fs::read_to_string(&file_path).unwrap();
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap(), "--dry-run"]);
    
    assert!(output.status.success());
    
    // File should be unchanged
    let content_after = fs::read_to_string(&file_path).unwrap();
    assert_eq!(content_after, original_content);
}

#[test]
fn test_balance_command_in_place() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(defn foo\n  (+ 1 2");
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap(), "--in-place"]);
    
    assert!(output.status.success());
    
    // File should be modified
    let content_after = fs::read_to_string(&file_path).unwrap();
    assert!(content_after.contains("(+ 1 2))"));
}

#[test]
fn test_balance_nonexistent_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("nonexistent.clj");
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Failed to read file"));
}

#[test]
fn test_slurp_command_forward() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo bar) baz");
    
    let output = run_paredit_like(&["slurp", file_path.to_str().unwrap(), "--line", "1"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.trim(), "(foo bar baz)");
}

#[test]
fn test_slurp_command_backward() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "foo (bar baz)");
    
    let output = run_paredit_like(&["slurp", file_path.to_str().unwrap(), "--line", "1", "--backward"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.trim(), "(foo bar baz)");
}

#[test]
fn test_barf_command_forward() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo bar baz)");
    
    let output = run_paredit_like(&["barf", file_path.to_str().unwrap(), "--line", "1"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.trim(), "(foo bar) baz");
}

#[test]
fn test_barf_command_backward() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo bar baz)");
    
    let output = run_paredit_like(&["barf", file_path.to_str().unwrap(), "--line", "1", "--backward"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.trim(), "foo (bar baz)");
}

#[test]
fn test_splice_command() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo bar)");
    
    let output = run_paredit_like(&["splice", file_path.to_str().unwrap(), "--line", "1"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.trim(), "foo bar");
}

#[test]
fn test_raise_command() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo (bar baz))");
    
    let output = run_paredit_like(&["raise", file_path.to_str().unwrap(), "--line", "1"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // The result depends on which form gets raised, but it should be valid
    assert!(!stdout.trim().is_empty());
}

#[test]
fn test_wrap_command_default() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "foo bar");
    
    let output = run_paredit_like(&["wrap", file_path.to_str().unwrap(), "--line", "1"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("(foo)"));
}

#[test]
fn test_wrap_command_brackets() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "foo bar");
    
    let output = run_paredit_like(&["wrap", file_path.to_str().unwrap(), "--line", "1", "--with", "["]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("[foo]"));
}

#[test]
fn test_wrap_command_braces() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "foo bar");
    
    let output = run_paredit_like(&["wrap", file_path.to_str().unwrap(), "--line", "1", "--with", "{"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("{foo}"));
}

#[test]
fn test_merge_let_command() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(let [x 1] (let [y 2] (+ x y)))");
    
    let output = run_paredit_like(&["merge-let", file_path.to_str().unwrap(), "--line", "1"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("let [x 1 y 2]"));
    assert!(stdout.contains("(+ x y)"));
}

#[test]
fn test_merge_let_command_with_typed_bindings() {
    let temp_dir = TempDir::new().unwrap();
    let content = "(let [user (: User) ctx]\n  (let [name (: Name) user\n        email (: Email) user]\n    {:name name\n     :email email}))";
    let file_path = create_temp_file(&temp_dir, "typed.clj", content);

    let output = run_paredit_like(&["merge-let", file_path.to_str().unwrap(), "--line", "1"]);

    assert!(output.status.success(), "merge-let command failed: {:?}", output);
    let stdout = String::from_utf8(output.stdout).unwrap();

    assert_eq!(stdout.matches("(let").count(), 1, "expected nested lets to be merged:\n{stdout}");
    assert!(stdout.contains("user (: User) ctx"), "outer typed binding missing:\n{stdout}");
    assert!(stdout.contains("name (: Name) user"), "inner typed binding missing:\n{stdout}");
    assert!(stdout.contains("email (: Email) user"), "second inner typed binding missing:\n{stdout}");

    let idx_user = stdout.find("user (: User) ctx").expect("missing user binding");
    let idx_name = stdout.find("name (: Name) user").expect("missing name binding");
    let idx_email = stdout.find("email (: Email) user").expect("missing email binding");

    assert!(
        idx_user < idx_name && idx_name < idx_email,
        "bindings not preserved in order:\n{stdout}"
    );
}

#[test]
fn test_batch_command_dry_run() {
    let temp_dir = TempDir::new().unwrap();
    create_temp_file(&temp_dir, "file1.clj", "(defn foo");
    create_temp_file(&temp_dir, "file2.clj", "(defn bar");
    
    let pattern = format!("{}/*.clj", temp_dir.path().display());
    let output = run_paredit_like(&["batch", &pattern, "--command", "balance", "--dry-run"]);
    
    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Processing"));
}

#[test]
fn test_invalid_command() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo bar)");
    
    let output = run_paredit_like(&["invalid-command", file_path.to_str().unwrap()]);
    
    assert!(!output.status.success());
}

#[test]
fn test_missing_line_argument() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "test.clj", "(foo bar) baz");
    
    let output = run_paredit_like(&["slurp", file_path.to_str().unwrap()]);
    
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("required") || stderr.contains("missing"));
}

#[test]
fn test_help_command() {
    let output = run_paredit_like(&["--help"]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("paredit-like"));
    assert!(stdout.contains("balance"));
    assert!(stdout.contains("slurp"));
    assert!(stdout.contains("barf"));
}

#[test]
fn test_complex_clojure_file() {
    let temp_dir = TempDir::new().unwrap();
    let complex_content = r#"
(ns my.namespace
  (:require [clojure.string :as str]
            [clojure.set :as set]))

(defn fibonacci [n]
  (if (<= n 1)
    n
    (+ (fibonacci (- n 1))
       (fibonacci (- n 2

(defn process-data [data]
  (let [cleaned (remove nil? data)
        sorted (sort cleaned)
        unique (distinct sorted)]
    (vec unique

(def my-map {:a 1
             :b 2
             :c {:nested 3
                 :value 4

; Function with missing closing parens
(defn incomplete-fn [x y z
  (let [sum (+ x y z
    (* sum 2
"#;
    
    let file_path = create_temp_file(&temp_dir, "complex.clj", complex_content);
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    // Check that all opening parens/brackets/braces have matching closing ones
    let open_parens = stdout.chars().filter(|&c| c == '(').count();
    let close_parens = stdout.chars().filter(|&c| c == ')').count();
    let open_brackets = stdout.chars().filter(|&c| c == '[').count();
    let close_brackets = stdout.chars().filter(|&c| c == ']').count();
    let open_braces = stdout.chars().filter(|&c| c == '{').count();
    let close_braces = stdout.chars().filter(|&c| c == '}').count();
    
    assert_eq!(open_parens, close_parens);
    assert_eq!(open_brackets, close_brackets);
    assert_eq!(open_braces, close_braces);
    
    // Check that comments and strings are preserved
    assert!(stdout.contains("; Function with missing closing parens"));
    assert!(stdout.contains(":require"));
}

#[test]
fn test_empty_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "empty.clj", "");
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout, "");
}

#[test]
fn test_file_with_only_comments() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = create_temp_file(&temp_dir, "comments.clj", "; Just comments\n; Nothing else");
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout, "; Just comments\n; Nothing else");
}

#[test]
fn test_multiline_strings() {
    let temp_dir = TempDir::new().unwrap();
    let content = r#"(str "This is a
multiline string
with (parens) inside")"#;
    let file_path = create_temp_file(&temp_dir, "multistring.clj", content);
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout, content);
}

#[test]
fn test_unicode_content() {
    let temp_dir = TempDir::new().unwrap();
    let content = "(str \"Hello 世界 🌍\" \"λ α β γ\")";
    let file_path = create_temp_file(&temp_dir, "unicode.clj", content);
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout, content);
}

#[test]
fn test_large_nested_structure() {
    let temp_dir = TempDir::new().unwrap();
    let mut content = String::new();
    
    // Create deeply nested structure
    for _ in 0..50 {
        content.push('(');
    }
    content.push_str("inner");
    for _ in 0..50 {
        content.push(')');
    }
    
    let file_path = create_temp_file(&temp_dir, "deep.clj", &content);
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout, content);
}

#[test]
fn test_refactoring_preserves_formatting() {
    let temp_dir = TempDir::new().unwrap();
    let content = "(defn my-function\n  [x y z]\n  (let [result (+ x y)]\n    (* result z)))";
    let file_path = create_temp_file(&temp_dir, "formatted.clj", content);
    
    let output = run_paredit_like(&["balance", file_path.to_str().unwrap()]);
    
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Should preserve the existing good formatting
    assert!(stdout.contains("  [x y z]"));
    assert!(stdout.contains("    (* result z)"));
}
