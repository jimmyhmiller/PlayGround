# TODO
- [x] Audit `/Users/jimmyhmiller/Documents/Code/open-source/git-of-theseus` to catalog features, data outputs, and CLI surface. (done via CLI read-through)
- [x] Define Rust project structure and seed an improved CLI skeleton (Clap-based multi-command).
- [x] Rename the Rust crate/binary to `git-history-visualizer` for a distinct identity from the Python tool.
- [ ] Port repository analysis engine (`git-of-theseus-analyze`) to Rust, matching JSON outputs against Python oracle. (basic parity established on toy repo; extend coverage and edge cases)
- [x] Finalize file filtering logic using the generated Pygments pattern set and globbing semantics.
- [ ] Port plotting commands (`stack-plot`, `line-plot`, `survival-plot`) using Rust-native charting.
- [x] Build regression tests comparing Rust outputs with the Python reference on sample repos. (see `tests/analysis_parity.rs`)
- [ ] Polish CLI ergonomics, packaging, and documentation.
