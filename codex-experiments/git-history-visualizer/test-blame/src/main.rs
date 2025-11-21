use std::path::Path;
use gix::bstr::ByteSlice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: test-blame <repo-path> <file-path> [commit]");
        std::process::exit(1);
    }

    let repo_path = &args[1];
    let file_path = &args[2];

    println!("Opening repository: {}", repo_path);
    let repo = gix::open(repo_path)?;

    let commit_id = if args.len() >= 4 {
        println!("Using provided commit: {}", &args[3]);
        gix::ObjectId::from_hex(args[3].as_bytes())?
    } else {
        println!("Getting HEAD commit...");
        let head = repo.head()?.peel_to_commit_in_place()?;
        head.id
    };

    println!("Blaming file: {} at commit {}", file_path, commit_id);

    let blame_result = repo.blame_file(
        file_path.as_bytes().as_bstr(),
        commit_id,
        gix::repository::blame_file::Options {
            follow_renames: true,
            ..Default::default()
        },
    )?;

    println!("\n=== BLAME RESULT ===");
    println!("Number of entries: {}", blame_result.entries.len());

    let mut total_lines = 0;
    for (i, entry) in blame_result.entries.iter().enumerate() {
        let range = entry.range_in_blamed_file();
        let lines = range.end - range.start;
        total_lines += lines;

        if i < 10 {
            println!("  Entry {}: lines {}..{} ({} lines) -> commit {}",
                i, range.start, range.end, lines, entry.commit_id);
        }
    }

    if blame_result.entries.len() > 10 {
        println!("  ... and {} more entries", blame_result.entries.len() - 10);
    }

    println!("\nTotal lines blamed: {}", total_lines);
    println!("File blob size: {} bytes", blame_result.blob.len());

    // Count actual lines in the blob
    let actual_lines = blame_result.blob.split(|&b| b == b'\n').count();
    println!("Actual lines in blob: {}", actual_lines);

    // Compare with git blame at the same commit
    println!("\n=== Comparing with git blame ===");
    let commit_str = format!("{}", commit_id);
    let output = std::process::Command::new("git")
        .args(&["-C", repo_path, "blame", &commit_str, "--", file_path])
        .output()?;

    if output.status.success() {
        let git_blame_lines = output.stdout.split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .count();
        println!("git blame line count: {}", git_blame_lines);

        if git_blame_lines != total_lines {
            println!("❌ MISMATCH: gix-blame={}, git blame={}, diff={}",
                total_lines, git_blame_lines, (git_blame_lines as i64) - (total_lines as i64));
        } else {
            println!("✅ MATCH: Both report {} lines", total_lines);
        }
    } else {
        eprintln!("Failed to run git blame: {}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
}
