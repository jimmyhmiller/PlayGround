use std::collections::HashMap;
use bumpalo::Bump;
use crate::{Term, TreeTerm, AST, ScopeId, ScopeSet, BindingId, BindingInfo, ParseError, Associativity, MacroTransformer};

pub mod binding_env;
pub mod enforester;
pub mod two_pass;

pub use binding_env::*;
pub use enforester::*;
pub use two_pass::*;

pub struct Parser<'arena> {
    pub enforester: Enforester<'arena>,
    pub arena: &'arena Bump,
}

impl<'arena> Parser<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            enforester: Enforester::new(arena),
            arena,
        }
    }

    pub fn parse(&mut self, terms: Vec<Term>) -> Result<Vec<&'arena AST<'arena>>, ParseError> {
        let (tree_terms, bindings) = self.parse1(terms, ScopeId(0))?;
        self.parse2(tree_terms, bindings)
    }
}