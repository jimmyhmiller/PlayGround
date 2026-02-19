pub mod term;
pub mod pattern;
pub mod scope;
pub mod parser;
pub mod engine;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;

pub use term::{TermId, SymId, VarId, TermData, TermStore};
pub use pattern::{Pattern, Env, Clause, Rule, match_pattern, substitute, pattern_to_term};
pub use scope::ScopeHandler;
pub use parser::{Token, Lexer, Parser, Program};
pub use engine::Engine;
