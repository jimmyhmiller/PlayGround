//! `defmmio-reg` (lib/mmio.coil): memory-mapped device registers as a typed bitfield
//! — a PURE MACRO (the §7 capstone). These host-level tests prove the macro expands
//! to valid, compiling code (no qemu needed); the bare-metal RUN is in
//! tests/freestanding.rs (gated). The cardinal — that this is a pure macro with NO
//! core feature — is verifiable by grep: nothing in src/ handles MMIO/registers.

const PROG: &str = "(module app)\n\
    (import \"lib/mmio.coil\" :use *)\n\
    (defmmio-reg UARTFR 150994968 [(rxfe 4 1) (txff 5 1)])\n";

#[test]
fn defmmio_reg_expands_to_reader_writer_and_field_getters() {
    // One register spec generates: a volatile-load reader, a volatile-store writer,
    // and a shift+mask getter PER field — all as ordinary defns.
    let src = format!("{PROG}(defn main [] (-> i64) 0)");
    let expanded = coil::expand_to_string(&src).expect("expand");
    for needed in [
        "(defn UARTFR-read",            // the reader
        "(defn UARTFR-write",           // the writer
        "(defn UARTFR-rxfe",            // field getter (bit 4)
        "(defn UARTFR-txff",            // field getter (bit 5)
        "load volatile i32",            // reader uses a volatile MMIO load
        "(ishr (UARTFR-read) 5)",       // txff shifts by its bit offset
    ] {
        assert!(expanded.contains(needed), "expansion missing {needed:?}:\n{expanded}");
    }
}

#[test]
fn defmmio_reg_output_typechecks_and_compiles() {
    // The generated accessors are real, valid Coil — they type-check and lower to LLVM
    // (a volatile load from a fixed address + shift/mask), with no core support.
    let src = format!("{PROG}(defn main [] (-> i64) (cast i64 (UARTFR-txff)))");
    assert!(coil::check_source(&src).is_ok(), "generated code must type-check");
    assert!(coil::emit_ir(&src).is_ok(), "generated code must reach codegen");
}
