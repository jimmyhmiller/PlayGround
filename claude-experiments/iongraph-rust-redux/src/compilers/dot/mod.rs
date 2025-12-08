pub mod schema;
pub mod parser;
pub mod ir_impl;

pub use schema::{DotGraph, DotNode, DotEdge};
pub use parser::{parse_dot, ParseError};
pub use ir_impl::dot_to_universal;
