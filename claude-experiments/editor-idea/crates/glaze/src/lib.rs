//! # Glaze
//!
//! A staged style/shader language. This crate is the **pure compiler** — no Bevy,
//! no rendering — so it can be developed and tested in complete isolation from the
//! host application (mirrors `editor_core`).
//!
//! Pipeline today (Stage 0 + Stage 1 slice):
//! `lex → parse → resolve(tokens, variant, states) → CompiledStyle`.
//!
//! Stages 2/3 (variant/state generalization is partly here; binding-time analysis
//! + WGSL backend for `shader{}` layers) are not implemented yet — a style with a
//! shader layer returns a hard [`GlazeError::Unsupported`] rather than being
//! silently dropped.
//!
//! ```
//! use glaze::parse;
//! use std::collections::HashMap;
//!
//! let prog = parse(r#"
//!     token accent.solid   = oklch(0.72 0.11 85)
//!     token surface.raised = oklch(0.28 0.01 250)
//!     token radius.md      = 8px
//!
//!     style button(intent) {
//!         fill   intent == danger ? oklch(0.58 0.16 25) : accent.solid
//!         pad    8px 12px
//!         radius radius.md
//!         :focus { border accent.solid 2px }
//!     }
//! "#).unwrap();
//!
//! let mut v = HashMap::new();
//! v.insert("intent".into(), "primary".into());
//! let compiled = prog.resolve("button", &v, &[]).unwrap();
//! assert_eq!(compiled.box_.radius, 8.0);
//! assert_eq!(compiled.layers.len(), 1); // one fill
//! ```

pub mod ast;
pub mod eval;
pub mod lexer;
pub mod parser;
pub mod shader;

pub use ast::{Expr, FnDef, Item, Program, ShaderBody, StyleDef, TokenDef};
pub use eval::{BoxStyle, CompiledStyle, Dim, Dir, Layer, Length, Rgba, Value};
pub use parser::parse;
pub use shader::{Builtin, CompiledShader};

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum GlazeError {
    Lex(String),
    Parse(String),
    Eval(String),
    /// A valid construct that the current stage cannot compile yet. Surfaced
    /// loudly (never a silent no-op) so the host can render a clear error.
    Unsupported(String),
}

impl fmt::Display for GlazeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlazeError::Lex(m) => write!(f, "glaze lex error: {m}"),
            GlazeError::Parse(m) => write!(f, "glaze parse error: {m}"),
            GlazeError::Eval(m) => write!(f, "glaze eval error: {m}"),
            GlazeError::Unsupported(m) => write!(f, "glaze: not implemented yet: {m}"),
        }
    }
}

impl std::error::Error for GlazeError {}

impl From<std::num::ParseIntError> for GlazeError {
    fn from(e: std::num::ParseIntError) -> Self {
        GlazeError::Eval(format!("bad integer: {e}"))
    }
}
