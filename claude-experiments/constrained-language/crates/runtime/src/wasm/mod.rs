//! WASM-component handler bodies.
//!
//! The top-level entry point is [`load_handler_body`], which takes a
//! manifest plus a handler name plus a path to a `.wasm` component and
//! returns a `BodyFn` the runtime can register against the handler's URI.

pub mod loader;
pub mod values;

pub use loader::{load_handler_body, LoadError};
pub use values::{json_to_val, val_to_json, ConvertError};
