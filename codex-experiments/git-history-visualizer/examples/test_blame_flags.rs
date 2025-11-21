/// Test different git2 blame flags to understand behavior
use git2::{Repository, BlameOptions};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <repo_path> <file_path>", args[0]);
        std::process::exit(1);
    }

    let repo = Repository::open(&args[1]).expect("Failed to open repo");
    let commit = repo.head().unwrap().peel_to_commit().unwrap();
    let file_path = &args[2];

    println!("Testing blame options on {} at commit {}\n", file_path, commit.id());

    // Test 1: Default options with newest_commit
    let mut opts1 = BlameOptions::new();
    opts1.newest_commit(commit.id());
    if let Ok(blame) = repo.blame_file(Path::new(file_path), Some(&mut opts1)) {
        let total: usize = blame.iter().map(|h| h.lines_in_hunk()).sum();
        println!("Default + newest_commit: {} lines", total);
    }

    // Test 2: With track_copies_same_file
    let mut opts2 = BlameOptions::new();
    opts2.newest_commit(commit.id());
    opts2.track_copies_same_file(true);
    if let Ok(blame) = repo.blame_file(Path::new(file_path), Some(&mut opts2)) {
        let total: usize = blame.iter().map(|h| h.lines_in_hunk()).sum();
        println!("With track_copies_same_file: {} lines", total);
    }

    // Test 3: With track_copies_any_commit_copies
    let mut opts3 = BlameOptions::new();
    opts3.newest_commit(commit.id());
    opts3.track_copies_any_commit_copies(true);
    if let Ok(blame) = repo.blame_file(Path::new(file_path), Some(&mut opts3)) {
        let total: usize = blame.iter().map(|h| h.lines_in_hunk()).sum();
        println!("With track_copies_any_commit_copies: {} lines", total);
    }

    // Test 4: Just the file without newest_commit
    let mut opts4 = BlameOptions::new();
    if let Ok(blame) = repo.blame_file(Path::new(file_path), Some(&mut opts4)) {
        let total: usize = blame.iter().map(|h| h.lines_in_hunk()).sum();
        println!("Without newest_commit: {} lines", total);
    }
}
