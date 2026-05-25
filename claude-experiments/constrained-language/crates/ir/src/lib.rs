//! IR for the constrained-language runtime.
//!
//! The IR is the single typed manifest every other component consumes: the
//! runtime loads it, the validator checks it, the WIT generator produces
//! per-handler component worlds from it, and the inspector renders it.

pub mod access;
pub mod canonical;
pub mod load;
pub mod manifest;
pub mod schema;
pub mod validate;
pub mod wit;

pub use access::{AccessPath, AccessSegment, KeyBinding, ParseError};
pub use load::{load_manifest_file, parse_json, parse_toml, LoadError};
pub use manifest::{ComponentRef, EffectDef, EventDef, Handler, Manifest, StateDecl};
pub use schema::{SchemaDef, SchemaRef};
pub use validate::{Issue, ValidationError};
