# gix-blame Bug Investigation

## Problem Statement
The Rust git-history-visualizer has a **20.26% undercount bug** where it reports 3,943,330 lines instead of the expected 4,945,213 lines (missing 1,001,883 lines).

## Timeline

### Initial State
- Using gix 0.70 with transitive dependency on gix-blame 0.0.0
- Performance: 77 seconds (4.9x faster than Python's 380 seconds)
- Accuracy: 20.7% undercount
- Many files returning 0 lines blamed even though `git blame` works correctly

### Attempted Fix: Upgrade to Latest gix
**Date**: 2025-11-16

**Changes Made**:
1. **Cargo.toml** (line 14):
   ```toml
   # FROM:
   gix = { version = "0.70", default-features = false, features = ["max-performance-safe", "revision", "index", "blob-diff", "mailmap", "blame"] }

   # TO:
   gix = { git = "https://github.com/GitoxideLabs/gitoxide", default-features = false, features = ["max-performance-safe", "revision", "index", "blob-diff", "mailmap", "blame"] }
   ```
   This pulled in gix v0.74.1 and gix-blame v0.4.0 (up from v0.0.0)

2. **src/analysis/mod.rs** (lines 708-713):
   ```rust
   // BEFORE: Manual topological traversal
   let mut diff_platform = context.repo.diff_resource_cache_for_tree_diff()?;
   use gix::prelude::*;
   let commits: Vec<_> = gix::traverse::commit::topo::Builder::from_iters(
       &context.repo.objects,
       [commit_id],
       None::<Vec<gix::ObjectId>>,
   )
   .build()?
   .collect();

   let blame_result = gix::blame::file(
       &context.repo.objects,
       commits,
       &mut diff_platform,
       file_path.as_ref(),
       None,
   );

   // AFTER: Using Repository convenience method
   let blame_result = context.repo.blame_file(
       file_path.as_ref(),
       commit_id,
       gix::repository::blame_file::Options::default(),
   );
   ```

3. **src/analysis/mod.rs** (lines 329 & 801):
   ```rust
   // Fixed signature handling for new API
   let resolved = if let Some(map) = mailmap {
       map.resolve(signature).to_owned()
   } else {
       signature.to_owned().expect("Failed to parse signature")  // Added .expect()
   };
   ```

**Result**:
- ✅ Build succeeded
- ✅ Performance maintained: 38 seconds (10x faster than Python)
- ❌ **Bug still present**: 3,943,330 lines (20.26% undercount, same as before)

## Current Status: BUG NOT FIXED

**Comparison**:
```
Python baseline:     4,945,213 lines
Rust (gix v0.74.1):  3,943,330 lines
Difference:          1,001,883 lines (20.26% undercount)
```

## Hypothesis
The initial hypothesis about a tokenizer bug in gix-blame v0.0.0 (lines 96-98 of function.rs that returns empty outcome when `num_lines_in_blamed == 0`) appears to be incorrect or incomplete. The 20% undercount persists even with gix-blame v0.4.0, suggesting:

1. The bug still exists in gix-blame v0.4.0
2. Our usage of the gix-blame API is incorrect
3. The issue is in our aggregation logic, not gix-blame itself
4. There's a different bug entirely

## Next Steps for Debugging

### 1. Identify Problem Files
```bash
# Compare which files have different line counts between Python and Rust
python3 << 'EOF'
import json

with open('/tmp/bench-python-playground/survival.json') as f:
    python_survival = json.load(f)
with open('/tmp/test-gix-new-full/survival.json') as f:
    rust_survival = json.load(f)

# Find files with mismatches
for file_hash in python_survival:
    python_lines = python_survival[file_hash][-1][1] if python_survival[file_hash] else 0
    rust_lines = rust_survival.get(file_hash, [[0, 0]])[-1][1]
    if python_lines != rust_lines:
        print(f"{file_hash}: Python={python_lines}, Rust={rust_lines}, Diff={python_lines - rust_lines}")
EOF
```

### 2. Test gix-blame Directly on Problem Files
Create a minimal test case that uses gix-blame directly on files that show discrepancies:
```rust
// Test rust/hello_cargo/Cargo.lock which was previously returning 0 lines
let outcome = repo.blame_file(
    "rust/hello_cargo/Cargo.lock".as_ref(),
    head_commit_id,
    Options::default()
)?;
println!("Entries: {}", outcome.entries.len());
for entry in &outcome.entries {
    let range = entry.range_in_blamed_file();
    println!("  Lines {}..{} ({} lines)", range.start, range.end, range.end - range.start);
}
```

### 3. Compare with git blame
```bash
# Get ground truth from git
git blame rust/hello_cargo/Cargo.lock | wc -l

# Compare with what gix-blame returns
```

### 4. Review Our Aggregation Logic
Check `src/analysis/mod.rs` around lines 714-730 where we process blame results:
- Are we correctly summing up all blame entries?
- Are we handling the ranges correctly?
- Could there be an off-by-one error?

### 5. Enable Debug Logging
The gix-blame Options struct has a `debug_track_path` field that might provide useful output:
```rust
let options = gix::repository::blame_file::Options {
    debug_track_path: true,
    ..Default::default()
};
```

### 6. Check for Deleted/Renamed Files
The Python implementation might be tracking files through renames while gix-blame isn't. Check the `rewrites` option in `gix::repository::blame_file::Options`.

## Test Files

### Example: rust/hello_cargo/Cargo.lock
- Location: `/Users/jimmyhmiller/Documents/Code/PlayGround/rust/hello_cargo/Cargo.lock`
- Expected lines: 23 (verified with `wc -l`)
- Previous behavior (gix 0.70): Returned 0 lines blamed
- Current behavior (gix 0.74.1): Unknown - needs verification

## References

- gix-blame source: https://github.com/GitoxideLabs/gitoxide/tree/main/gix-blame
- gix Repository::blame_file: `/tmp/gitoxide/gix/src/repository/blame.rs`
- Our implementation: `src/analysis/mod.rs` lines 708-730
