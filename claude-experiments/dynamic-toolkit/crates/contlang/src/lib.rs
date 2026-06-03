mod lexer;
pub mod lower;
mod parser;

pub use lexer::{Token, TokenKind, lex};
pub use lower::lower_program;
pub use parser::{Decl, Expr, Param, Program, ValType, parse};

#[cfg(test)]
mod tests;
