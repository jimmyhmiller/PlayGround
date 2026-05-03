//! Built-in dynamic-language stdlib types.
//!
//! Frontends will all need a "growable indexed sequence" — JS arrays,
//! Lua array-tables, Lox arrays, beagle arrays. dynobj provides the
//! `varlen_values` and `Raw64` field-kind primitives, but assembling a
//! sequence type from them is the same dance every time.
//! [`IndexedSeq`] does it once.

pub mod indexed_seq;

pub use indexed_seq::{IndexedSeq, SeqView};
