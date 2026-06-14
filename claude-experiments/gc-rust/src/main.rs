//! gc-rust compiler driver (`gcr`).
//!
//! Phase 0: a placeholder that proves the crate links against the GC runtime.
//! The real driver (parse → typecheck → monomorphize → codegen → run) lands in
//! later phases.

fn main() {
    eprintln!("gcr: gc-rust compiler (phase 0 — GC substrate online)");
}
