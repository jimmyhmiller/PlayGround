//! Authentication: the SCRAM-SHA-256 cryptographic core and (later) the user
//! verifier store glue. The wire framing that drives these state machines lives
//! in [`crate::protocol`]; the per-user verifier persistence lives on
//! [`crate::db::Database`].
pub mod scram;
