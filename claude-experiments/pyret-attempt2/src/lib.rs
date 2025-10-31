//! Pyret Parser - A Rust implementation of the Pyret language parser
//!
//! This library provides a complete parser for the Pyret programming language,
//! generating JSON AST that matches the reference JavaScript implementation.
//!
//! # Example
//!
//! ```no_run
//! use pyret_attempt2::{Tokenizer, Parser};
//!
//! let source = r#"
//! fun factorial(n):
//!   if n == 0: 1
//!   else: n * factorial(n - 1)
//!   end
//! end
//! "#;
//!
//! let mut tokenizer = Tokenizer::new(source);
//! let tokens = tokenizer.tokenize();
//!
//! let mut parser = Parser::new(tokens, "test.arr".to_string());
//! let program = parser.parse_program().unwrap();
//!
//! let json = serde_json::to_string_pretty(&program).unwrap();
//! println!("{}", json);
//! ```

pub mod ast;
pub mod error;
pub mod parser;
pub mod tokenizer;

// Re-export main types
pub use ast::*;
pub use error::{ParseError, ParseResult};
pub use parser::Parser;
pub use tokenizer::Tokenizer;
