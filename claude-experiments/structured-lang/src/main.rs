use std::io::{self, BufRead, Write};

use structured_lang::database::Database;
use structured_lang::wire::dispatch;

fn main() {
    let mut db = Database::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    eprintln!("structured-lang: JSON command REPL (one JSON object per line)");
    eprintln!("  ops: create_branch, fork_branch, create_table, add_column, insert_row, ...");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = dispatch(&mut db, trimmed);
        let json = serde_json::to_string(&response).unwrap_or_else(|e| {
            format!(r#"{{"ok":false,"error":"serialization error: {}"}}"#, e)
        });
        writeln!(stdout, "{}", json).ok();
        stdout.flush().ok();
    }
}
