//! Inspector views over an IR manifest.
//!
//! Each public function returns a `String` so that:
//! * the `main` binary just prints them, and
//! * tests can snapshot the exact output.

pub mod views;

pub use views::{handler_card, program_map, state_cell, validate_report};
