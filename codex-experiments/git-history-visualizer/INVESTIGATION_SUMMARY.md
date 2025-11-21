# Investigation Summary: 20% Undercount Bug

## Problem
When analyzing the PlayGround repository (~1977 commits, ~1811 files):
- Python (pygit2/libgit2): **4,945,213 total lines**
- Rust (gix-blame): **3,943,330 total lines**
- **Missing: 1,001,883 lines (20.26% undercount)**

## Investigation Findings

### 1. gix-blame Core Functionality is CORRECT ✅
- Created test tool `test-blame` to compare gix-blame with `git blame` on individual files
- **Result**: gix-blame produces IDENTICAL output to `git blame` for individual files at specific commits
- Example: `minimal-typechecker.ts` at commit `0c4c0428` - both report exactly 156 lines
- **Conclusion**: The bug is NOT in gix-blame's core algorithm

### 2. The "Unprocessed Hunks" Bug is REAL but NOT the Cause ✅
- Bug fixed in `gitoxide/gix-blame/src/file/function.rs` lines 402-426
- Fix handles remaining unblamed hunks at end of algorithm
- Debug output (lines 407, 410, 415, 422) NEVER appears during analysis
- **Conclusion**: This code path is never executed, so this bug doesn't cause the 20% undercount

### 3. Rename Tracking Makes it WORSE ❌
- Hypothesis: Missing rename tracking causes discrepancies
- Test: Enabled `rewrites` with `copies` in blame options
- **Result**: Line count DROPPED from 3.9M to 505K (made it 10x worse!)
- **Conclusion**: Rename tracking is NOT the solution

### 4. Minimal Test Repo Shows NO Bug ✅
- Created `/tmp/minimal-blame-test` with 24 commits, 27 lines total
- Rust analyzer correctly reports 27 lines
- **Conclusion**: The bug is triggered by specific patterns in large repos, not simple cases

### 5. Very Few Files Return 0 Lines
- Only 6 files out of 3,247 return 0 lines blamed
- Not enough to account for 1M missing lines
- **Conclusion**: The bug is not about files failing to blame entirely

## Current Hypothesis

The 20% undercount is likely caused by one of these scenarios:

1. **Different API defaults between libgit2 and gix**
   - libgit2 (Python baseline) may have different default options
   - Possible differences in: whitespace handling, rename detection, merge handling

2. **Subtle differences in commit attribution**
   - Some lines are attributed to different commits between the two implementations
   - This could cause double-counting in some commits and under-counting in others
   - Net effect: 20% undercount overall

3. **Edge case in historical file tracking**
   - The bug manifests when analyzing entire repository history
   - May be related to: deleted files, file renames across history, merge commits
   - Simple test repos don't trigger the issue

## What We Know for Sure

✅ gix-blame works correctly for individual file blames
✅ The unprocessed hunks bug exists but isn't executed
✅ Simple repos analyze correctly
✅ The bug appears in complex, real-world repositories
❌ Rename tracking is NOT the solution
❌ It's NOT about files failing to blame
❌ It's NOT a simple off-by-one error

## Next Steps to Find Root Cause

Since I cannot create a minimal reproducer, the bug requires deeper investigation:

1. **Compare Python vs Rust implementations line-by-line**
   - Check if Python's pygit2 uses different default options
   - Verify whitespace handling, rename tracking defaults, merge handling

2. **Trace specific problematic commits**
   - Use `find-discrepancy.py` to find commits with large discrepancies
   - Manually trace why those commits have different line counts
   - Check what `git blame` actually returns for those cases

3. **Check libgit2 vs gix behavioral differences**
   - libgit2 documentation for blame options
   - gix documentation for blame options
   - Identify any semantic differences

4. **Instrument the code**
   - Add detailed logging to track every blame call
   - Log the commit ID, file path, and line count for each blame
   - Compare logs between Python and Rust runs

## Conclusion

The root cause is NOT:
- A bug in gix-blame's core algorithm (verified to work correctly)
- The unprocessed hunks bug (code path not executed)
- Missing rename tracking (makes it worse)
- A simple implementation bug (minimal repo works fine)

The root cause IS likely:
- A subtle difference in how libgit2 and gix handle specific edge cases
- Different default options or behavior between the two libraries
- Something that only manifests in complex, real-world repository histories

**This requires access to the Python implementation or more detailed comparison of the two libraries' behavior.**
