# End-to-end bundler scaling

Diffpack uses one specialized incremental graph engine: a dense integer module
graph with forward and reverse adjacency lists and a maintained reachability
spanning tree.

Initial reachability is a cache-friendly dense traversal. Adding an edge
activates only newly reachable descendants. Removing a non-tree edge requires
no reachability work. Removing a tree edge detaches its subtree, searches for
alternative incoming edges from the reachable graph, and reactivates only the
surviving region. Large retractions adaptively use a complete dense traversal.

## Commands

```console
cargo build --release
target/release/diffpack bundle-scale-direct 10000 4
target/release/diffpack bundle-scale-direct-deps 10000 4
target/release/diffpack bundle-scale-direct-live 10000 4
target/release/diffpack bundle-scale-direct-live-deps 10000 4
```

The benchmark includes corpus generation as a separately reported field and
times discovery, Oxc transformation, resolution, reachability, rendering, and
one edit. The live variants make every module contribute to an observable entry
value and execute the resulting bundle as a correctness gate.

## Reachability behavior

On the synthetic dependency-removal workload, content-only edits take nearly no
reachability work because their edge delta is empty. Dependency deletion repairs
only the affected region unless it crosses the adaptive full-traversal
threshold. Tests compare the maintained result with a fresh traversal and cover
detached cycles, alternative incoming edges, and non-tree edge removal.

End-to-end matched results against Rolldown are maintained in
[ROLLDOWN_COMPARISON.md](ROLLDOWN_COMPARISON.md).

## Current bottlenecks

1. Cold discovery reads, parses, transforms, and resolves many small files.
2. Development mode still rewrites affected output instead of serving cached
   module artifacts over HMR.
3. The linker IR needs to become structured enough to build a combined Oxc AST
   for general compression and mangling.
4. Real application fixtures are more valuable than further optimization of
   synthetic reachability microbenchmarks.
