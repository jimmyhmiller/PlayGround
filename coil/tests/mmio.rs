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
fn defmmio_reg_is_general_not_pl011_specific() {
    // The capability is GENERAL — it works on ANY register spec, with multi-bit
    // fields (mask != 1), not just the PL011 flag register. A made-up GPIO control
    // register at a different address with fields of width 1, 2, and 3.
    let src = "(module app)\n\
        (import \"lib/mmio.coil\" :use *)\n\
        (defmmio-reg GPIOCTL 268435456 [(enable 0 1) (mode 1 2) (speed 4 3)])\n\
        (defn main [] (-> i64) (cast i64 (GPIOCTL-mode)))";
    let expanded = coil::expand_to_string(src).expect("expand");
    // each field shifts by its own offset and masks by (1<<width)-1 (1, 3, 7):
    assert!(expanded.contains("(ishr (GPIOCTL-read) 0)") && expanded.contains("(isub (ishl 1 1) 1)"),
        "enable: shift 0, mask (1<<1)-1:\n{expanded}");
    assert!(expanded.contains("(ishr (GPIOCTL-read) 1)") && expanded.contains("(isub (ishl 1 2) 1)"),
        "mode: shift 1, mask (1<<2)-1 = 3:\n{expanded}");
    assert!(expanded.contains("(ishr (GPIOCTL-read) 4)") && expanded.contains("(isub (ishl 1 3) 1)"),
        "speed: shift 4, mask (1<<3)-1 = 7:\n{expanded}");
    // and the whole thing type-checks + compiles for this different spec.
    assert!(coil::emit_ir(src).is_ok(), "GPIO spec must compile too");
}

#[test]
fn defmmio_reg_with_no_fields_is_read_write_only() {
    // An empty field list (a plain data register like UARTDR) generates EXACTLY a
    // reader + writer and no getters — and compiles.
    let src = "(module app)\n\
        (import \"lib/mmio.coil\" :use *)\n\
        (defmmio-reg UARTDR 150994944 [])\n\
        (defn main [] (-> i64) (do (UARTDR-write (cast u32 65)) (cast i64 (UARTDR-read))))";
    let expanded = coil::expand_to_string(src).expect("expand");
    assert!(expanded.contains("(defn UARTDR-read") && expanded.contains("(defn UARTDR-write"));
    assert_eq!(
        expanded.matches("(defn UARTDR-").count(),
        2,
        "empty field list must generate exactly read + write (no getters):\n{expanded}"
    );
    assert!(coil::emit_ir(src).is_ok());
}

#[test]
fn defmmio_reg_output_typechecks_and_compiles() {
    // The generated accessors are real, valid Coil — they type-check and lower to LLVM
    // (a volatile load from a fixed address + shift/mask), with no core support.
    let src = format!("{PROG}(defn main [] (-> i64) (cast i64 (UARTFR-txff)))");
    assert!(coil::check_source(&src).is_ok(), "generated code must type-check");
    assert!(coil::emit_ir(&src).is_ok(), "generated code must reach codegen");
}
