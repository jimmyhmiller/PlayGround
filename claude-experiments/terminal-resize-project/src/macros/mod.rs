use std::sync::Arc;
use crate::{Term, LiteralValue, BracketType, ParseError, MacroTransformer, Span};

pub mod and_or;
pub mod while_loop;
pub mod cond;

pub use and_or::*;
pub use while_loop::*;
pub use cond::*;

pub fn register_builtin_macros() -> Vec<(&'static str, Arc<dyn MacroTransformer>)> {
    vec![
        ("and", Arc::new(AndMacro)),
        ("or", Arc::new(OrMacro)),
        ("while", Arc::new(WhileMacro)),
        ("cond", Arc::new(CondMacro)),
    ]
}

fn expand_to_if(condition: Term, then_branch: Term, else_branch: Option<Term>) -> Term {
    let else_term = else_branch.unwrap_or(Term::Literal(LiteralValue::Nil));
    
    Term::Bracket(BracketType::Paren, vec![
        Term::Identifier("if".to_string(), Span::synthetic()),
        condition,
        then_branch,
        else_term,
    ])
}