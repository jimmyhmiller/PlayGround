//! Dogfood #4 (examples/cinterop.coil): the C-interop / systems-integration
//! capstone — Coil living in the C ecosystem. Calls real libc/libm across the C
//! ABI #5 built: qsort with a Coil COMPARATOR CALLBACK (C calls back into Coil),
//! libm sqrt/pow (float ABI), libc div (struct returned by value), strtol (char*).

mod common;
use common::build_and_run;

#[test]
fn calls_real_libc_libm_including_a_coil_callback() {
    let src = std::fs::read_to_string("examples/cinterop.coil").expect("read examples/cinterop.coil");
    // 42 iff qsort (via the Coil callback) sorted correctly AND sqrt/pow/strtol/div
    // all agree on 42 — i.e. the FFI callback, float ABI, struct-by-value return,
    // and char* args all work together against the system C libraries.
    assert_eq!(build_and_run(&src), 42);
}
