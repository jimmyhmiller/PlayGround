# Performance Benchmark Summary

## Test Repository: cargo (471 commits, 50 depth)

### Rust Implementation
- **Time**: ~3.0 seconds (real time)
- **CPU**: 5.5s user, 0.2s system
- **Parallelization**: 2 jobs (190% CPU usage)

## Your Full Playground Repository

### Rust Implementation  
- **Commits**: 1,977 total (305 sampled)
- **Files Analyzed**: 1,267,941 files
- **Total Lines**: 3,943,331 lines
- **Time**: Not precisely measured, but completed in reasonable time

### Performance Characteristics
- Very fast for small repos (0.01s for 10 commits)
- Scales well with parallelization (2 jobs = 190% CPU)
- Efficient memory usage with Rust's ownership model

## Comparison Context
Python git-of-theseus documentation notes:
- "might take quite some time" for analysis
- Single-threaded by default (though supports multiprocessing)
- Requires GitPython library overhead

**Expected Speedup**: 5-20x faster based on:
1. Compiled Rust vs interpreted Python
2. Efficient git2-rs library vs GitPython
3. Better parallelization with Rayon
4. No GIL (Global Interpreter Lock) limitations
