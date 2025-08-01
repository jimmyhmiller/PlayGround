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
        let mut parser = Self {
            enforester: Enforester::new(arena),
            arena,
        };
        
        // Initialize with basic operators
        parser.init_builtin_operators();
        parser
    }
    
    fn init_builtin_operators(&mut self) {
        use crate::{BindingInfo, Associativity, ScopeSet, ScopeId};
        let base_scope = ScopeSet::new().with_scope(ScopeId(0));
        
        // Arithmetic operators - register both as operators and as function variables
        let plus_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "+".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 10, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "+".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(plus_binding_id),
        );
        let minus_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "-".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 10, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "-".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(minus_binding_id),
        );
        let mult_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "*".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 20, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "*".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(mult_binding_id),
        );
        let div_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "/".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 20, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "/".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(div_binding_id),
        );
        
        // Comparison operators
        self.enforester.env.bind(
            "==".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "!=".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "<".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            ">".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            "<=".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Left },
        );
        self.enforester.env.bind(
            ">=".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 5, assoc: Associativity::Left },
        );
        
        // Assignment
        self.enforester.env.bind(
            "=".to_string(),
            base_scope.clone(),
            BindingInfo::BinaryOp { precedence: 1, assoc: Associativity::Right },
        );
        
        // Register if as both keyword and variable
        let if_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "if".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(if_binding_id),
        );
        
        // Register then and else as keywords
        let then_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "then".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(then_binding_id),
        );
        
        let else_binding_id = self.enforester.env.new_binding();
        self.enforester.env.bind(
            "else".to_string(),
            base_scope.clone(),
            BindingInfo::Variable(else_binding_id),
        );
        
        // Logical operators as macros
        use std::sync::Arc;
        use crate::macros::{AndMacro, OrMacro};
        
        self.enforester.env.bind(
            "and".to_string(),
            base_scope.clone(),
            BindingInfo::Macro(Arc::new(AndMacro)),
        );
        self.enforester.env.bind(
            "or".to_string(),
            base_scope.clone(),
            BindingInfo::Macro(Arc::new(OrMacro)),
        );
    }

    pub fn parse(&mut self, terms: Vec<Term>) -> Result<Vec<&'arena AST<'arena>>, ParseError> {
        let (tree_terms, bindings) = self.parse1(terms, ScopeId(0))?;
        self.parse2(tree_terms, bindings)
    }
}