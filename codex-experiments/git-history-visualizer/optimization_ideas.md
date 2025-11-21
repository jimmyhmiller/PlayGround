# Optimization Ideas for git-history-visualizer

## Current Performance Issues
- Rust uses 263s CPU vs Python's 174s (52% MORE CPU time!)
- But Rust has 12s sys vs Python's 130s (90% LESS system time)
- Suggests: We're doing more work but with better parallelization

## Potential Optimizations

### 1. Reduce Cloning (Memory & CPU)
- Line 306: `entry.path.clone()` 
- Line 312: `key.clone()` in hot loop
- Line 315, 318: `entry.clone()`
- Line 323: Unnecessary `Vec` collection
- Line 349: More key cloning

### 2. Better Parallelization
- Currently using 2 jobs - could auto-detect CPU count
- Thread pool might be underutilized

### 3. Git Operations
- Opening repo multiple times per thread
- Could reuse blame options

### 4. Algorithm Improvements
- Fast path for unchanged files could skip more work
- Better caching of file states

### 5. Data Structure Improvements
- Use `Rc` or `Arc` for shared strings instead of cloning
- Use string interning for repeated keys
