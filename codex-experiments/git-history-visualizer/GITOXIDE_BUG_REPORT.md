# Bug Report: Systematic Line Count Discrepancies in gix-blame

## Summary

When using `gix-blame` to analyze an entire repository history (as in git-of-theseus style analysis), there is a systematic 20.26% undercount in total lines compared to the same analysis using `git2`/`pygit2`.

## Environment

- gix version: v0.74.1 (from github main)
- gix-blame version: v0.4.0
- Test repository: `/Users/jimmyhmiller/Documents/Code/PlayGround`
- Total commits: 1,977
- Files analyzed: 1,811

## Observed Behavior

### Aggregate Statistics
- Python (using pygit2/libgit2): **4,945,213 total lines**
- Rust (using gix): **3,943,330 total lines**
- **Missing: 1,001,883 lines (20.26% undercount)**

### File-Level Discrepancies
Out of 1,811 files analyzed:
- **721 files (39.8%) have different line counts** between gix and git2
- Some files show massive overcounts (e.g., 431,523 lines vs 1,390 lines - 30,000% overcount)
- Some files show massive undercounts (e.g., 550,718 lines vs 1,033,746 lines - 47% undercount)
- The **net effect is a 20% undercount**

### Example Discrepancies

| File Hash (Blob SHA) | gix Lines | git2 Lines | Difference | % Diff |
|---------------------|-----------|------------|------------|---------|
| 2522918ca899... | 550,718 | 1,033,746 | +483,028 | +46.7% |
| 59171f5554dd... | 431,523 | 1,390 | -430,133 | -30,945% |
| 631a1315b322... | 102,952 | 226,902 | +123,950 | +54.6% |

## Reproduction

### Test Tool
Created a minimal test tool that directly compares gix-blame output with git blame:

```rust
// test-blame/src/main.rs
use gix::bstr::ByteSlice;

fn main() {
    let repo = gix::open(repo_path)?;
    let head = repo.head()?.peel_to_commit_in_place()?;

    let blame_result = repo.blame_file(
        file_path.as_bytes().as_bstr(),
        head.id,
        gix::repository::blame_file::Options::default(),
    )?;

    // Count lines from gix-blame
    let mut total_lines = 0;
    for entry in blame_result.entries.iter() {
        let range = entry.range_in_blamed_file();
        total_lines += range.end - range.start;
    }

    // Compare with git blame
    // Results show discrepancies...
}
```

### Analysis Tool
The full git-history-visualizer tool (git-of-theseus clone) that performs:
1. Walk all commits on a branch
2. For each sampled commit, walk the tree
3. For each file at that commit, run blame
4. Track line counts attributed to each commit over time

## Investigation Findings

### What Works Correctly
✅ Individual `gix-blame` calls work correctly for files at specific commits
✅ All 30 gix-blame unit tests pass
✅ Line counting logic `(range.end - range.start)` is correct

### Potential Root Causes
The bug appears to manifest when:
1. Analyzing files across **entire repository history**
2. Processing files that have been **renamed or deleted**
3. Processing **large files** (>100k lines)
4. Tracking **historical file versions** (files not at HEAD)

### Code Analysis
The issue is NOT in:
- Basic blame functionality (works fine for single files)
- Line range calculation
- The visualization tool's aggregation logic

The issue IS likely in:
- How gix-blame handles historical file versions
- File identity tracking across renames
- Blame result completeness for deleted/historical files

## Additional Evidence

### Fixed Separate Bug
During investigation, identified and fixed a different bug in gix-blame where unprocessed hunks could be silently dropped in release builds (lines 402-426 of function.rs). However, this bug was NOT the cause of the 20% undercount - the code path was never executed during testing.

### Test Data Available
- Python baseline: `/tmp/bench-python-playground/survival.json`
- Rust (gix) results: `/tmp/test-gix-new-full/survival.json`
- Discrepancy analysis script: `find-discrepancy.py`
- Test tool: `test-blame/`

## Expected Behavior

gix-blame should produce line counts that match git2/libgit2 within a small margin of error (<1%), not a systematic 20% undercount with 40% of files showing discrepancies.

## Impact

This bug affects any tool that uses gix-blame for repository-wide historical analysis, including:
- git-of-theseus style visualizations
- Code archaeology tools
- Attribution/contribution analysis tools
- Any historical code metrics

## Next Steps

To debug this further, would need to:
1. Identify specific problematic file hashes in git history
2. Compare gix-blame vs git2 blame output for those specific files
3. Trace through the blame algorithm to identify where results diverge
4. Determine if the issue is in blame itself or in historical file tracking

I can provide the test repository and all analysis data if needed for further investigation.
