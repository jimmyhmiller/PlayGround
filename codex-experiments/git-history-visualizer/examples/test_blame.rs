use git2::{Repository, BlameOptions};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <repo_path>", args[0]);
        std::process::exit(1);
    }

    let repo_path = &args[1];
    let repo = Repository::open(repo_path).expect("Failed to open repo");
    let commit = repo.head().unwrap().peel_to_commit().unwrap();

    let mut options = BlameOptions::new();
    options.newest_commit(commit.id());

    let file_path = if args.len() > 2 {
        &args[2]
    } else {
        "file.txt"
    };

    let blame = repo.blame_file(Path::new(file_path), Some(&mut options))
        .expect("Failed to blame file");

    println!("Blaming {} at commit {}", file_path, commit.id());
    println!("Total hunks: {}", blame.len());

    let mut total_lines = 0;
    for (i, hunk) in blame.iter().enumerate() {
        let lines = hunk.lines_in_hunk();
        total_lines += lines;
        println!("Hunk {}: lines_in_hunk={}, final_commit={}, orig_start_line={}, final_start_line={}",
            i,
            lines,
            hunk.final_commit_id(),
            hunk.orig_start_line(),
            hunk.final_start_line()
        );
    }

    println!("\nTotal lines counted: {}", total_lines);
}
