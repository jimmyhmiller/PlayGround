mod lexer;
mod parser;
mod lower;
pub mod segment_interp;
pub mod generic_interp;
pub mod unified_interp;

pub use lexer::{lex, Token, TokenKind};
pub use parser::{parse, Expr, Decl, Param, Program, ValType};
pub use lower::lower_program;

#[cfg(test)]
mod tests;
