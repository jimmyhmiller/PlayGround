use std::collections::BTreeSet;
use std::sync::Arc;
use bumpalo::Bump;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
    
    pub fn synthetic() -> Self {
        Self { start: 0, end: 0 }
    }
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

#[derive(Debug, Clone)]
pub enum BracketType {
    Paren,   // ()
    Square,  // []
    Curly,   // {}
}

// Phase 1: Raw terms after reading
#[derive(Debug, Clone)]
pub enum Term {
    Identifier(String, Span),
    Literal(LiteralValue),
    Bracket(BracketType, Vec<Term>),
    Operator(String, Span),
}

// Scope management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScopeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingId(pub u32);

#[derive(Debug, Clone)]
pub struct ScopeSet {
    pub scopes: BTreeSet<ScopeId>,
}

impl ScopeSet {
    pub fn new() -> Self {
        ScopeSet {
            scopes: BTreeSet::new(),
        }
    }

    pub fn with_scope(mut self, scope: ScopeId) -> Self {
        self.scopes.insert(scope);
        self
    }

    pub fn subset_of(&self, other: &ScopeSet) -> bool {
        self.scopes.is_subset(&other.scopes)
    }
}

// Phase 2: After enforestation
#[derive(Debug)]
pub enum TreeTerm<'arena> {
    Literal(LiteralValue),
    Variable(&'arena str, ScopeSet),
    BinaryOp {
        op: &'arena str,
        left: &'arena TreeTerm<'arena>,
        right: &'arena TreeTerm<'arena>,
        span: Span,
    },
    Block {
        terms: Vec<Term>,
        scope_id: ScopeId,
    },
    MacroCall {
        name: &'arena str,
        args: Vec<&'arena TreeTerm<'arena>>,
        span: Span,
    },
    Function {
        name: &'arena str,
        params: Vec<&'arena str>,
        body: &'arena TreeTerm<'arena>,
    },
    If {
        condition: &'arena TreeTerm<'arena>,
        then_branch: &'arena TreeTerm<'arena>,
        else_branch: Option<&'arena TreeTerm<'arena>>,
    },
    Call {
        func: &'arena TreeTerm<'arena>,
        args: Vec<&'arena TreeTerm<'arena>>,
    },
}

// Phase 3: After full expansion and binding resolution
#[derive(Debug)]
pub enum AST<'arena> {
    ResolvedVar {
        name: &'arena str,
        binding_id: BindingId,
    },
    Literal(LiteralValue),
    FunctionCall {
        func: &'arena AST<'arena>,
        args: Vec<&'arena AST<'arena>>,
    },
    FunctionDef {
        binding_id: BindingId,
        params: Vec<BindingId>,
        body: &'arena AST<'arena>,
    },
    Block(Vec<&'arena AST<'arena>>),
    If {
        condition: &'arena AST<'arena>,
        then_branch: &'arena AST<'arena>,
        else_branch: Option<&'arena AST<'arena>>,
    },
    Loop {
        body: &'arena AST<'arena>,
    },
    Break,
}

#[derive(Debug, Clone)]
pub enum Associativity {
    Left,
    Right,
}

// Binding information
#[derive(Debug, Clone)]
pub enum BindingInfo<'arena> {
    Variable(BindingId),
    Macro(Arc<dyn MacroTransformer + 'arena>),
    BinaryOp { precedence: u32, assoc: Associativity },
}

pub trait MacroTransformer: Send + Sync {
    fn transform(&self, terms: Vec<Term>) -> Result<Vec<Term>, ParseError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Unexpected end of input")]
    UnexpectedEndOfInput,
    
    #[error("Unexpected operator: {0}")]
    UnexpectedOperator(String),
    
    #[error("Unexpected term")]
    UnexpectedTerm,
    
    #[error("Extra terms in parentheses")]
    ExtraTermsInParens,
    
    #[error("Unbound variable: {0}")]
    UnboundVariable(String),
    
    #[error("Expected keyword: {0}")]
    ExpectedKeyword(String),
    
    #[error("Invalid macro pattern")]
    InvalidMacroPattern,
    
    #[error("Macro expansion error: {0}")]
    MacroExpansionError(String),
    
    #[error("Unexpected tree term")]
    UnexpectedTreeTerm,
    
    #[error("Lexer error: {0}")]
    LexerError(String),
}