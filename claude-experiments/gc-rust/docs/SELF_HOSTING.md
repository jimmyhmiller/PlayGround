# Self-hosting — dropped (proof-of-concept)

gc-rust briefly had a self-hosting bootstrap compiler (`compiler/*.gcr`) that
reached a stage2==stage3 fixpoint, emitting i64-uniform code linked against a
separate non-moving C runtime (`runtime/gcr_rt.c`).

**This was a proof of concept and has been removed.** gc-rust is a Rust + inkwell
implementation; self-hosting is **not** a goal. The precise, moving, generational
GC and JVM-grade tooling are the focus — see [`FUTURE_WORK.md`](FUTURE_WORK.md).

(The bootstrap was never committed to git; if it is ever needed again it must be
recovered from a backup, not from history.)
