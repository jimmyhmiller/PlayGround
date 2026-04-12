mod lexer;
mod parser;
pub mod lower;

pub use lexer::{lex, Token, TokenKind};
pub use parser::{parse, Expr, Decl, Param, Program, ValType};
pub use lower::lower_program;

#[cfg(test)]
mod tests;
