# Implementation Status: Rename Tracking Fix

## Summary
Root cause identified: **gix-blame doesn't follow file renames by default**, causing lines to be attributed to move commits instead of the commits that originally added them. This accounts for the 20% undercount bug.

## What Was Completed ✅

### Phase 1: Add `follow_renames` Option (COMPLETE)
**Files modified:**
1. `gitoxide/gix-blame/src/types.rs` - Added `follow_renames: bool` field to Options struct
2. `gitoxide/gix/src/repository/mod.rs` - Added `follow_renames: bool` to public API Options
3. `gitoxide/gix/src/repository/blame.rs` - Pass through `follow_renames` option

**Status**: All option plumbing is in place and compiles

### Phase 2: Implement `build_rename_history` Function (PARTIAL)
**File modified:**
- `gitoxide/gix-blame/src/file/function.rs` - Added `build_rename_history()` function after line 920

**Status**: Function skeleton is written but has API mismatches that need fixing

**Issues to resolve:**
- Need to use correct gix API calls (e.g., `find_commit` from `gix_traverse::commit::find`)
- Need to properly handle commit/tree decoding with buffers
- Need to match existing code patterns more closely

### Phase 3: Integrate into Main Loop (PARTIAL)
**File modified:**
- `gitoxide/gix-blame/src/file/function.rs` - Added rename_map building (lines 117-134) and usage (lines 149-157)

**Status**: Integration logic is in place but won't compile until Phase 2 is fixed

## What Remains ❌

### Fix `build_rename_history` Implementation
The function needs to be rewritten to match gix's API patterns:

```rust
// Example of correct API usage (from existing code):
use gix_traverse::commit::find as find_commit;

let commit = find_commit(cache.as_ref(), &odb, &commit_id, &mut buf)?;
let tree_id = commit.tree_id()?;
let tree = odb.find_tree(&tree_id, &mut buf2)?;
```

Key corrections needed:
1. Use `find_commit()` from gix_traverse instead of `odb.find_commit()`
2. Pass buffers to all find operations (avoid allocations)
3. Use `.tree_id()` method on CommitRef, not `.tree_id()` on object
4. Remove `MissingTree` error variant (doesn't exist)
5. Fix `tree_diff_at_file_path` call signature (needs all 12 parameters)

### Phase 4: Add Tests
**Not started** - Need to:
1. Add test case for chain of renames (A→B→C)
2. Add test for single rename
3. Verify behavior matches git blame

### Phase 5: Update Analyzer
**File to modify:**
- `src/analysis/mod.rs` - Enable `follow_renames: true` in blame options

**Simple change:**
```rust
let blame_options = gix::repository::blame_file::Options {
    follow_renames: true,
    ..Default::default()
};
```

### Phase 6: Validation
1. Test with `onebigfile.xml` (should attribute to commit `301e1475` not `59171f55`)
2. Run full PlayGround analysis (should get ~4.9M lines vs current 3.9M)
3. Compare survival.json with Python baseline

## Alternative Approach: Simpler Workaround

Instead of fixing gix-blame itself, we could implement rename tracking in the analyzer:

**Pros:**
- Doesn't require modifying gitoxide
- Can use existing git commands (`git log --follow`)
- Faster to implement

**Cons:**
- Performance overhead (extra git process per file)
- Not a proper fix to gix-blame
- Doesn't help other gix-blame users

**Implementation:**
```rust
// Before blaming, find historical path
fn find_file_at_commit(repo_path: &Path, commit: &str, file: &str) -> Result<String> {
    let output = Command::new("git")
        .args(&["-C", repo_path.to_str().unwrap(), "log", "--follow",
                "--format=%H", "--name-only", commit, "--", file])
        .output()?;
    // Parse to find file path at that commit
    ...
}
```

## Proof of Root Cause

**Test case:** `rust/asm-arm2/resources/onebigfile.xml` (430K lines)

**Commits:**
- `301e1475` - ADDED file (April 9, 2023)
- `59171f55` - MOVED file to subdirectory (May 4, 2023)

**Current behavior (WRONG):**
```bash
$ test-blame /path/to/repo rust/asm-arm2/asm/resources/onebigfile.xml
Entry 0: lines 0..430082 (430082 lines) -> commit 59171f55
```
❌ Attributes all 430K lines to the MOVE commit

**Expected behavior:**
```bash
$ git blame 59171f55 -- rust/asm-arm2/asm/resources/onebigfile.xml | awk '{print $1}' | sort | uniq -c
  430082 301e1475
```
✅ git blame correctly attributes to the ADD commit

**Impact:** This single file accounts for 43% of the 1M missing lines!

## Files Changed So Far

1. ✅ `gitoxide/gix-blame/src/types.rs` - Add option
2. ✅ `gitoxide/gix/src/repository/mod.rs` - Add option
3. ✅ `gitoxide/gix/src/repository/blame.rs` - Pass option
4. ⚠️  `gitoxide/gix-blame/src/file/function.rs` - Add rename tracking (HAS COMPILE ERRORS)

## Next Steps

**Option A: Fix the gix-blame implementation**
1. Fix `build_rename_history()` API calls to match gix patterns
2. Test compilation
3. Add tests
4. Enable in analyzer
5. Validate results

**Option B: Use workaround in analyzer**
1. Implement git log --follow wrapper
2. Build rename map before analysis
3. Pass correct historical paths to gix-blame
4. Validate results

**Recommendation**: Try Option B first as a quick validation that this approach works, then pursue Option A as a proper long-term fix to contribute to gitoxide.
