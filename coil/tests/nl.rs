//! Dogfood #3 (examples/nl.coil): the alloc/IO-INTERFACE capstone — battle-tests
//! the capability-as-a-value design by implementing a CUSTOM Reader backend (over
//! an in-memory slice) and a CUSTOM Writer that WRAPS another Writer (composition),
//! threading an explicit allocator + IO through a real `nl` (number lines) tool.

mod common;
use common::build_and_capture;

#[test]
fn nl_numbers_lines_via_custom_reader_and_wrapping_writer() {
    let src = std::fs::read_to_string("examples/nl.coil").expect("read examples/nl.coil");
    let (code, out) = build_and_capture(&src);
    assert_eq!(out, "1  alpha\n2  beta\n3  gamma\n"); // numbered via the wrapping count-Writer
    assert_eq!(code, 3); // line count, read through the custom slice-Reader
}
