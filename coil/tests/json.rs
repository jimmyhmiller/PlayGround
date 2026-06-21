//! Dogfood #2: the JSON parser + recursive value model (examples/json.coil) — the
//! Phase-2 capstone exercising (slice u8) strings + str-keyops, a recursive sum
//! with collection pointers, ArrayList<Json> + HashMap<(slice u8) Json>, recursion,
//! references, and the control-flow macros all together. We compile+run the actual
//! example: it parses a nested object/array document and queries values out of it.

mod common;
use common::build_and_run;

#[test]
fn json_example_parses_and_queries() {
    let src = std::fs::read_to_string("examples/json.coil").expect("read examples/json.coil");
    // nums[0]+nums[2] (10+12) + meta.k (40) - nums[1] (20) = 42 — exercises array
    // indexing, nested object lookup (str-keyops), and number parsing end-to-end.
    assert_eq!(build_and_run(&src), 42);
}
