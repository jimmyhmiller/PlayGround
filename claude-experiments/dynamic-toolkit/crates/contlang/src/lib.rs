mod lexer;
mod parser;
mod lower;
pub mod unified_interp;
pub mod gc_stack;

pub use lexer::{lex, Token, TokenKind};
pub use parser::{parse, Expr, Decl, Param, Program, ValType};
pub use lower::lower_program;

#[cfg(test)]
mod tests;
