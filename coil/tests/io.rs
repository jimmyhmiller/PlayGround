//! Zig-style IO as a library: a Writer/Reader is an explicit capability value
//! (a vtable of function pointers) threaded through. Errors are a sum type
//! (Result), never a -1 sentinel. The same code runs against stdout, a fixed
//! memory buffer, or a null sink by swapping the value.

mod common;
use common::{build_and_capture, build_and_run};

#[test]
fn writes_to_stdout_through_a_writer() {
    let (code, out) = build_and_capture(include_str!("../examples/io.coil"));
    assert_eq!(code, 0);
    assert_eq!(out, "answer=42\n");
}

#[test]
fn print_int_formats_decimal() {
    let src = r#"
        (module test)
        (import "lib/io.coil" :use *)
        (import "lib/result.coil" :use *)
        (defn main [] (-> :i64)
          (do (print-int (stdout) 12345)
              (write-byte (stdout) (cast :u8 10))
              0))
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "12345\n");
}

#[test]
fn fixed_buffer_writer_captures_bytes() {
    // Format into an in-memory buffer (no syscalls), then read the length back.
    let src = r#"
        (module test)
        (import "lib/alloc.coil" :use *)
        (import "lib/io.coil" :use *)
        (import "lib/result.coil" :use *)
        (defn main [] (-> :i64)
          (let [a   (malloc-allocator)
                buf (unwrap-ptr (alloc-slice [i8] a 64))
                fb  (alloc-static FixBuf)]
            (store! (field fb data) buf)
            (store! (field fb cap) 64)
            (let [w (fixed-buffer-writer fb)]
              (print-str w "hello")          ; 5 bytes
              (print-int w 42)               ; 2 bytes
              (load (field fb len)))))        ; total written = 7
    "#;
    assert_eq!(build_and_run(src), 7);
}

#[test]
fn writer_is_polymorphic_over_the_sink() {
    // The same `emit` runs against a null sink (discards, always succeeds).
    let src = r#"
        (module test)
        (import "lib/io.coil" :use *)
        (import "lib/result.coil" :use *)
        (defn emit [(w (ptr Writer))] (-> (Result :i64 IoError))
          (do (print-str w "ignored") (print-int w 999)))
        (defn main [] (-> :i64)
          (match (emit (null-writer)) (Ok [n] 42) (Err [e] 0)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn io_is_visible_in_the_signature() {
    // A function that outputs must take a Writer; the capability is threaded in.
    let src = r#"
        (module test)
        (import "lib/io.coil" :use *)
        (import "lib/result.coil" :use *)
        (defn banner [(w (ptr Writer))] (-> (Result :i64 IoError))
          (print-str w "hi\n"))
        (defn main [] (-> :i64)
          (match (banner (stdout)) (Ok [n] 0) (Err [e] 1)))
    "#;
    let (code, out) = build_and_capture(src);
    assert_eq!(code, 0);
    assert_eq!(out, "hi\n");
}
