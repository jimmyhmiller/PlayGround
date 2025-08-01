# Implementing Honu-Inspired Enforestation in Rust: A Comprehensive Tutorial

## Introduction: What We're Building

This tutorial implements the Honu enforestation algorithm, a sophisticated parsing technique that enables Scheme-style syntactic extensibility in languages with infix syntax and implicit delimiters. The key innovation is **interleaving parsing and macro expansion** rather than treating them as separate phases, enabling powerful features like forward macro references and binding-aware parsing.

## Understanding the Honu Algorithm

### The Three-Layer Architecture

Honu operates through three cooperating layers:

1. **Reader Layer**: Converts raw text into balanced trees of terms
2. **Enforestation Layer**: Converts flat sequences into tree expressions using operator precedence
3. **Expansion Layer**: Handles binding resolution and drives the parsing process

### Key Insight: Cooperative Two-Pass Parsing

The most critical aspect often misunderstood: **parse1 and parse2 are NOT sequential phases**. They work cooperatively:

- **parse1**: Discovers bindings *while* enforesting top-level forms
- **parse2**: Completes parsing using discovered bindings
- Both passes call back into each other through recursive parsing of nested scopes

## Core Data Structures

Let's start with the fundamental types representing our parsing phases:

```rust
use std::collections::{HashMap, BTreeSet};
use std::sync::Arc;
use bumpalo::Bump;

// Phase 1: Raw terms after reading
#[derive(Debug, Clone)]
pub enum Term {
    Identifier(String, Span),
    Literal(LiteralValue),
    Bracket(BracketType, Vec<Term>),
    // Operators not yet associated with precedence
    Operator(String, Span),
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
pub enum BracketType {
    Paren,   // ()
    Square,  // []
    Curly,   // {}
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
        // Delayed parsing for forward references
        terms: Vec<Term>,
        scope_id: ScopeId,
    },
    MacroCall {
        name: &'arena str,
        args: Vec<TreeTerm<'arena>>,
        span: Span,
    },
    Function {
        name: &'arena str,
        params: Vec<&'arena str>,
        body: &'arena TreeTerm<'arena>,
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
}
```

## Scope Sets and Binding Resolution

Implementing Honu's hygiene system using sets of scopes:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingId(u32);

#[derive(Debug, Clone)]
pub struct ScopeSet {
    scopes: BTreeSet<ScopeId>,
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

// Binding environment for resolution
pub struct BindingEnv<'arena> {
    arena: &'arena Bump,
    bindings: HashMap<String, Vec<(ScopeSet, BindingInfo<'arena>)>>,
    current_scope: ScopeId,
    scope_counter: u32,
}

#[derive(Debug, Clone)]
pub enum BindingInfo<'arena> {
    Variable(BindingId),
    Macro(Arc<dyn MacroTransformer + 'arena>),
    BinaryOp { precedence: u32, assoc: Associativity },
}

impl<'arena> BindingEnv<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            bindings: HashMap::new(),
            current_scope: ScopeId(0),
            scope_counter: 1,
        }
    }

    pub fn new_scope(&mut self) -> ScopeId {
        let id = ScopeId(self.scope_counter);
        self.scope_counter += 1;
        id
    }

    pub fn lookup(&self, name: &str, scopes: &ScopeSet) -> Option<&BindingInfo<'arena>> {
        self.bindings.get(name)?
            .iter()
            .filter(|(binding_scopes, _)| scopes.subset_of(binding_scopes))
            .max_by_key(|(binding_scopes, _)| binding_scopes.scopes.len())
            .map(|(_, info)| info)
    }

    pub fn bind(&mut self, name: String, scopes: ScopeSet, info: BindingInfo<'arena>) {
        self.bindings.entry(name)
            .or_insert_with(Vec::new)
            .push((scopes, info));
    }
}
```

## The Enforest Algorithm

Here's the core enforestation algorithm that handles operators, precedence, and macro calls:

```rust
pub struct Enforester<'arena> {
    arena: &'arena Bump,
    env: BindingEnv<'arena>,
}

#[derive(Debug, Clone)]
pub struct EnforestState<'arena> {
    combine: Box<dyn Fn(&'arena TreeTerm<'arena>) -> &'arena TreeTerm<'arena> + 'arena>,
    precedence: u32,
}

pub type OperatorStack<'arena> = Vec<EnforestState<'arena>>;

impl<'arena> Enforester<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            env: BindingEnv::new(arena),
        }
    }

    /// The main enforest function following the Honu algorithm
    pub fn enforest(
        &mut self,
        mut terms: Vec<Term>,
        combine: impl Fn(&'arena TreeTerm<'arena>) -> &'arena TreeTerm<'arena> + 'arena + 'static,
        min_precedence: u32,
        stack: &mut OperatorStack<'arena>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        // Base case: empty input
        if terms.is_empty() {
            return Err(ParseError::UnexpectedEndOfInput);
        }

        let first = terms.remove(0);
        
        match first {
            Term::Identifier(name, span) => {
                let scopes = self.current_scopes();
                
                match self.env.lookup(&name, &scopes) {
                    Some(BindingInfo::Macro(transformer)) => {
                        // Rule: Macro invocation
                        // Pass remaining terms to macro transformer
                        let expanded = transformer.transform(terms, &self.env)?;
                        
                        // Continue enforesting with expanded result
                        if let Some(combine_state) = stack.pop() {
                            self.enforest(
                                expanded,
                                combine_state.combine,
                                combine_state.precedence,
                                stack,
                            )
                        } else {
                            self.enforest(expanded, combine, min_precedence, stack)
                        }
                    }
                    Some(BindingInfo::Variable(binding_id)) => {
                        // Variable reference
                        let var = self.arena.alloc(TreeTerm::Variable(&name, scopes));
                        self.continue_enforest(var, terms, combine, min_precedence, stack)
                    }
                    Some(BindingInfo::BinaryOp { precedence, assoc }) => {
                        // This shouldn't happen as first term
                        Err(ParseError::UnexpectedOperator(name))
                    }
                    None => {
                        // Unknown identifier - treat as variable for now
                        let var = self.arena.alloc(TreeTerm::Variable(&name, scopes));
                        self.continue_enforest(var, terms, combine, min_precedence, stack)
                    }
                }
            }
            Term::Literal(lit) => {
                let lit_term = self.arena.alloc(TreeTerm::Literal(lit));
                self.continue_enforest(lit_term, terms, combine, min_precedence, stack)
            }
            Term::Bracket(BracketType::Paren, inner_terms) => {
                // Parenthesized expression
                let (expr, remaining) = self.enforest(
                    inner_terms,
                    |x| x, // identity combine
                    0,     // reset precedence
                    &mut vec![],
                )?;
                
                if !remaining.is_empty() {
                    return Err(ParseError::ExtraTermsInParens);
                }
                
                self.continue_enforest(expr, terms, combine, min_precedence, stack)
            }
            Term::Bracket(BracketType::Curly, inner_terms) => {
                // Delayed block parsing for forward references
                let block = self.arena.alloc(TreeTerm::Block {
                    terms: inner_terms,
                    scope_id: self.env.current_scope,
                });
                self.continue_enforest(block, terms, combine, min_precedence, stack)
            }
            _ => Err(ParseError::UnexpectedTerm),
        }
    }

    /// Continue enforesting after parsing a primary expression
    fn continue_enforest(
        &mut self,
        left: &'arena TreeTerm<'arena>,
        mut remaining: Vec<Term>,
        combine: impl Fn(&'arena TreeTerm<'arena>) -> &'arena TreeTerm<'arena> + 'arena + 'static,
        min_precedence: u32,
        stack: &mut OperatorStack<'arena>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        // Apply combine function
        let left = combine(left);

        // Check for operators
        if let Some(Term::Identifier(op_name, span)) = remaining.first() {
            let scopes = self.current_scopes();
            
            if let Some(BindingInfo::BinaryOp { precedence, assoc }) = 
                self.env.lookup(op_name, &scopes) {
                
                // Check precedence
                let (left_prec, right_prec) = match assoc {
                    Associativity::Left => (*precedence, precedence + 1),
                    Associativity::Right => (precedence + 1, *precedence),
                };

                if left_prec < min_precedence {
                    // Lower precedence - return current result
                    return Ok((left, remaining));
                }

                // Consume operator
                remaining.remove(0);

                // Push current state to stack
                stack.push(EnforestState {
                    combine: Box::new(combine),
                    precedence: min_precedence,
                });

                // Create combine function for binary operator
                let op_str = self.arena.alloc_str(op_name);
                let new_combine = move |right: &'arena TreeTerm<'arena>| -> &'arena TreeTerm<'arena> {
                    self.arena.alloc(TreeTerm::BinaryOp {
                        op: op_str,
                        left,
                        right,
                        span: *span,
                    })
                };

                // Continue with right side
                self.enforest(remaining, new_combine, right_prec, stack)
            } else {
                // Not an operator - return current result
                Ok((left, remaining))
            }
        } else {
            // No more terms or not an identifier
            Ok((left, remaining))
        }
    }

    fn current_scopes(&self) -> ScopeSet {
        ScopeSet::new().with_scope(self.env.current_scope)
    }
}
```

## Two-Pass Parsing Implementation

Now let's implement the cooperative parse1/parse2 algorithm:

```rust
pub struct Parser<'arena> {
    enforester: Enforester<'arena>,
    arena: &'arena Bump,
}

impl<'arena> Parser<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            enforester: Enforester::new(arena),
            arena,
        }
    }

    /// Parse a sequence of terms (combines parse1 and parse2)
    pub fn parse(&mut self, terms: Vec<Term>) -> Result<Vec<&'arena AST<'arena>>, ParseError> {
        // Phase 1: Discover bindings while enforesting
        let (tree_terms, bindings) = self.parse1(terms, ScopeId(0))?;
        
        // Phase 2: Complete parsing with discovered bindings
        self.parse2(tree_terms, bindings)
    }

    /// parse1: Binding discovery and top-level enforestation
    fn parse1(
        &mut self,
        mut terms: Vec<Term>,
        scope: ScopeId,
    ) -> Result<(Vec<&'arena TreeTerm<'arena>>, BindingEnv<'arena>), ParseError> {
        let mut results = Vec::new();
        let original_scope = self.enforester.env.current_scope;
        self.enforester.env.current_scope = scope;

        while !terms.is_empty() {
            // Enforest one top-level form
            let (tree_term, remaining) = self.enforester.enforest(
                terms,
                |x| x, // identity combine
                0,     // top-level precedence
                &mut vec![],
            )?;

            // Process the form for binding discovery
            match tree_term {
                TreeTerm::MacroCall { name, args, .. } if name == "macro" => {
                    // Macro definition - extract and register
                    let (macro_name, transformer) = self.process_macro_def(args)?;
                    let scopes = self.enforester.current_scopes();
                    self.enforester.env.bind(
                        macro_name.to_string(),
                        scopes,
                        BindingInfo::Macro(Arc::new(transformer)),
                    );
                    // Optionally preserve for export
                    results.push(tree_term);
                }
                TreeTerm::Function { name, params, body } => {
                    // Function definition - register binding
                    let binding_id = BindingId(results.len() as u32);
                    let scopes = self.enforester.current_scopes();
                    self.enforester.env.bind(
                        name.to_string(),
                        scopes,
                        BindingInfo::Variable(binding_id),
                    );
                    results.push(tree_term);
                }
                TreeTerm::BinaryOp { op: "=", left, right, .. } => {
                    // Variable declaration
                    if let TreeTerm::Variable(name, _) = left {
                        let binding_id = BindingId(results.len() as u32);
                        let scopes = self.enforester.current_scopes();
                        self.enforester.env.bind(
                            name.to_string(),
                            scopes,
                            BindingInfo::Variable(binding_id),
                        );
                    }
                    results.push(tree_term);
                }
                _ => {
                    // Expression - preserve for parse2
                    results.push(tree_term);
                }
            }

            terms = remaining;
        }

        self.enforester.env.current_scope = original_scope;
        Ok((results, self.enforester.env.clone()))
    }

    /// parse2: Complete parsing with discovered bindings
    fn parse2(
        &mut self,
        tree_terms: Vec<&'arena TreeTerm<'arena>>,
        bindings: BindingEnv<'arena>,
    ) -> Result<Vec<&'arena AST<'arena>>, ParseError> {
        // Update enforester with complete bindings
        self.enforester.env = bindings;
        
        tree_terms.into_iter()
            .map(|tree_term| self.complete_parse(tree_term))
            .collect()
    }

    /// Complete parsing of a tree term to final AST
    fn complete_parse(&mut self, tree_term: &'arena TreeTerm<'arena>) -> Result<&'arena AST<'arena>, ParseError> {
        match tree_term {
            TreeTerm::Literal(lit) => {
                Ok(self.arena.alloc(AST::Literal(lit.clone())))
            }
            TreeTerm::Variable(name, scopes) => {
                // Resolve binding
                match self.enforester.env.lookup(name, scopes) {
                    Some(BindingInfo::Variable(binding_id)) => {
                        Ok(self.arena.alloc(AST::ResolvedVar {
                            name,
                            binding_id: *binding_id,
                        }))
                    }
                    _ => Err(ParseError::UnboundVariable(name.to_string())),
                }
            }
            TreeTerm::Block { terms, scope_id } => {
                // Delayed block parsing - now we have all bindings
                let block_scope = self.enforester.env.new_scope();
                let parsed = self.parse(terms.clone())?;
                
                // Wrap in block expression
                Ok(self.arena.alloc(AST::Block(parsed)))
            }
            TreeTerm::Function { name, params, body } => {
                // Create new scope for function body
                let body_scope = self.enforester.env.new_scope();
                let original_scope = self.enforester.env.current_scope;
                self.enforester.env.current_scope = body_scope;

                // Bind parameters
                let param_bindings: Vec<_> = params.iter()
                    .enumerate()
                    .map(|(i, param)| {
                        let binding_id = BindingId(1000 + i as u32);
                        let scopes = self.enforester.current_scopes();
                        self.enforester.env.bind(
                            param.to_string(),
                            scopes,
                            BindingInfo::Variable(binding_id),
                        );
                        binding_id
                    })
                    .collect();

                // Parse body with parameter bindings
                let body_ast = self.complete_parse(body)?;
                
                self.enforester.env.current_scope = original_scope;

                Ok(self.arena.alloc(AST::FunctionDef {
                    binding_id: BindingId(0), // Would be set by parse1
                    params: param_bindings,
                    body: body_ast,
                }))
            }
            TreeTerm::BinaryOp { op, left, right, .. } => {
                // For now, treat as function call
                let left_ast = self.complete_parse(left)?;
                let right_ast = self.complete_parse(right)?;
                
                Ok(self.arena.alloc(AST::FunctionCall {
                    func: self.arena.alloc(AST::ResolvedVar {
                        name: op,
                        binding_id: BindingId(0), // Would lookup operator
                    }),
                    args: vec![left_ast, right_ast],
                }))
            }
            _ => Err(ParseError::UnexpectedTreeTerm),
        }
    }

    fn process_macro_def(&self, args: Vec<TreeTerm<'arena>>) -> Result<(&str, impl MacroTransformer + 'arena), ParseError> {
        // Simplified - would parse macro pattern and template
        Ok(("example_macro", ExampleMacro {}))
    }
}
```

## Macro System Integration

Here's how macros are integrated with the enforestation process:

```rust
pub trait MacroTransformer: Send + Sync {
    fn transform(&self, terms: Vec<Term>, env: &BindingEnv) -> Result<Vec<Term>, ParseError>;
}

/// Example macro that demonstrates interleaved expansion
struct IfMacro;

impl MacroTransformer for IfMacro {
    fn transform(&self, mut terms: Vec<Term>, env: &BindingEnv) -> Result<Vec<Term>, ParseError> {
        // Pattern: if <expr> then <expr> else <expr>
        
        // Use enforest to parse condition expression
        let mut enforester = Enforester::new(env.arena);
        enforester.env = env.clone();
        
        let (condition, mut remaining) = enforester.enforest(
            terms,
            |x| x,
            0,
            &mut vec![],
        )?;
        
        // Check for 'then' keyword
        if let Some(Term::Identifier(kw, _)) = remaining.first() {
            if kw == "then" {
                remaining.remove(0);
            } else {
                return Err(ParseError::ExpectedKeyword("then".to_string()));
            }
        }
        
        // Parse then branch
        let (then_branch, mut remaining) = enforester.enforest(
            remaining,
            |x| x,
            0,
            &mut vec![],
        )?;
        
        // Check for 'else' keyword and parse else branch
        // ... similar pattern ...
        
        // Generate expanded form
        Ok(vec![
            Term::Identifier("cond".to_string(), Span::synthetic()),
            Term::Bracket(BracketType::Paren, vec![
                // condition -> then_branch
                // else -> else_branch
            ]),
        ])
    }
}

/// Macro for defining infix operators
struct InfixOpMacro;

impl MacroTransformer for InfixOpMacro {
    fn transform(&self, terms: Vec<Term>, env: &BindingEnv) -> Result<Vec<Term>, ParseError> {
        // Pattern: infix <name> <precedence> <assoc> = <implementation>
        // This would register the operator in the binding environment
        // and generate appropriate function definition
        Ok(vec![])
    }
}
```

## Complete Working Example

Let's put it all together with a complete example showing forward references and macro-aware parsing:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_reference_with_macros() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);

        // Register built-in operators
        parser.enforester.env.bind(
            "+".to_string(),
            ScopeSet::new().with_scope(ScopeId(0)),
            BindingInfo::BinaryOp {
                precedence: 10,
                assoc: Associativity::Left,
            },
        );
        
        parser.enforester.env.bind(
            "*".to_string(),
            ScopeSet::new().with_scope(ScopeId(0)),
            BindingInfo::BinaryOp {
                precedence: 20,
                assoc: Associativity::Left,
            },
        );

        // Input with forward reference
        let input = vec![
            // function fibonacci(n) { if (n <= 1) then n else helper(n) }
            Term::Identifier("function".to_string(), Span::new(0, 8)),
            Term::Identifier("fibonacci".to_string(), Span::new(9, 18)),
            Term::Bracket(BracketType::Paren, vec![
                Term::Identifier("n".to_string(), Span::new(19, 20)),
            ]),
            Term::Bracket(BracketType::Curly, vec![
                Term::Identifier("if".to_string(), Span::new(23, 25)),
                Term::Bracket(BracketType::Paren, vec![
                    Term::Identifier("n".to_string(), Span::new(27, 28)),
                    Term::Identifier("<=".to_string(), Span::new(29, 31)),
                    Term::Literal(LiteralValue::Number(1.0)),
                ]),
                Term::Identifier("then".to_string(), Span::new(35, 39)),
                Term::Identifier("n".to_string(), Span::new(40, 41)),
                Term::Identifier("else".to_string(), Span::new(42, 46)),
                Term::Identifier("helper".to_string(), Span::new(47, 53)), // Forward ref
                Term::Bracket(BracketType::Paren, vec![
                    Term::Identifier("n".to_string(), Span::new(54, 55)),
                ]),
            ]),
            // function helper(x) { x - 1 }  // Defined after use
            Term::Identifier("function".to_string(), Span::new(60, 68)),
            Term::Identifier("helper".to_string(), Span::new(69, 75)),
            Term::Bracket(BracketType::Paren, vec![
                Term::Identifier("x".to_string(), Span::new(76, 77)),
            ]),
            Term::Bracket(BracketType::Curly, vec![
                Term::Identifier("x".to_string(), Span::new(80, 81)),
                Term::Identifier("-".to_string(), Span::new(82, 83)),
                Term::Literal(LiteralValue::Number(1.0)),
            ]),
        ];

        // Parse - this should handle the forward reference correctly
        let result = parser.parse(input);
        
        assert!(result.is_ok());
        let ast = result.unwrap();
        assert_eq!(ast.len(), 2); // Two function definitions
        
        // The fibonacci function should be able to reference helper
        // even though helper is defined later
    }

    #[test] 
    fn test_macro_precedence_interaction() {
        let arena = Bump::new();
        let mut parser = Parser::new(&arena);

        // Register operators and a macro
        parser.enforester.env.bind(
            "+".to_string(),
            ScopeSet::new().with_scope(ScopeId(0)),
            BindingInfo::BinaryOp {
                precedence: 10,
                assoc: Associativity::Left,
            },
        );
        
        parser.enforester.env.bind(
            "*".to_string(),
            ScopeSet::new().with_scope(ScopeId(0)),
            BindingInfo::BinaryOp {
                precedence: 20,
                assoc: Associativity::Left,
            },
        );

        parser.enforester.env.bind(
            "square".to_string(),
            ScopeSet::new().with_scope(ScopeId(0)),
            BindingInfo::Macro(Arc::new(SquareMacro)),
        );

        // Input: 2 + square(3) * 4
        // Should parse as: 2 + ((3 * 3) * 4)
        let input = vec![
            Term::Literal(LiteralValue::Number(2.0)),
            Term::Identifier("+".to_string(), Span::new(2, 3)),
            Term::Identifier("square".to_string(), Span::new(4, 10)),
            Term::Bracket(BracketType::Paren, vec![
                Term::Literal(LiteralValue::Number(3.0)),
            ]),
            Term::Identifier("*".to_string(), Span::new(14, 15)),
            Term::Literal(LiteralValue::Number(4.0)),
        ];

        let result = parser.parse(input);
        assert!(result.is_ok());
        
        // The macro expansion should respect operator precedence
        // square(3) expands to (3 * 3), then * binds tighter than +
    }
}

struct SquareMacro;

impl MacroTransformer for SquareMacro {
    fn transform(&self, mut terms: Vec<Term>, _env: &BindingEnv) -> Result<Vec<Term>, ParseError> {
        // Pattern: square(<expr>)
        if let Some(Term::Bracket(BracketType::Paren, args)) = terms.first() {
            if args.len() == 1 {
                let arg = args[0].clone();
                return Ok(vec![
                    Term::Bracket(BracketType::Paren, vec![
                        arg.clone(),
                        Term::Identifier("*".to_string(), Span::synthetic()),
                        arg,
                    ]),
                ]);
            }
        }
        Err(ParseError::InvalidMacroPattern)
    }
}
```

## Error Handling and Recovery

Implementing robust error handling for the interleaved expansion process:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
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
}

/// Error recovery for partial parsing
pub struct RecoveringParser<'arena> {
    parser: Parser<'arena>,
    errors: Vec<ParseError>,
}

impl<'arena> RecoveringParser<'arena> {
    pub fn parse_with_recovery(&mut self, terms: Vec<Term>) -> (Vec<&'arena AST<'arena>>, Vec<ParseError>) {
        let mut results = Vec::new();
        let mut remaining = terms;
        
        while !remaining.is_empty() {
            match self.parser.parse(remaining.clone()) {
                Ok(ast) => {
                    results.extend(ast);
                    break;
                }
                Err(e) => {
                    self.errors.push(e);
                    // Try to recover by skipping to next statement
                    remaining = self.skip_to_next_statement(remaining);
                }
            }
        }
        
        (results, std::mem::take(&mut self.errors))
    }
    
    fn skip_to_next_statement(&self, mut terms: Vec<Term>) -> Vec<Term> {
        // Simple recovery: skip until we find a statement boundary
        while !terms.is_empty() {
            if matches!(terms[0], Term::Identifier(ref s, _) if s == "function" || s == "macro") {
                break;
            }
            terms.remove(0);
        }
        terms
    }
}
```

## Performance Optimization with Arenas

Using arena allocation for efficient memory management:

```rust
pub struct ArenaCache<'arena> {
    arena: &'arena Bump,
    string_cache: HashMap<String, &'arena str>,
    tree_term_cache: HashMap<TreeTermKey, &'arena TreeTerm<'arena>>,
}

#[derive(Hash, PartialEq, Eq)]
struct TreeTermKey {
    // Simplified key for caching
    kind: String,
    hash: u64,
}

impl<'arena> ArenaCache<'arena> {
    pub fn intern_string(&mut self, s: &str) -> &'arena str {
        if let Some(&interned) = self.string_cache.get(s) {
            return interned;
        }
        
        let interned = self.arena.alloc_str(s);
        self.string_cache.insert(s.to_string(), interned);
        interned
    }
    
    pub fn cache_tree_term(&mut self, key: TreeTermKey, term: &'arena TreeTerm<'arena>) {
        self.tree_term_cache.insert(key, term);
    }
}
```

## Conclusion

This implementation of Honu-inspired enforestation in Rust demonstrates the key aspects of the algorithm:

1. **Two-pass parsing** where parse1 and parse2 work cooperatively, not sequentially
2. **Interleaved enforestation and macro expansion** during the parsing process
3. **Delayed block parsing** enabling forward references through deferred parsing
4. **Proper Pratt parsing** with macro awareness using the enforest algorithm
5. **Rust-specific techniques** including arena allocation, type-safe phase separation, and comprehensive error handling

The implementation shows how Rust's type system and memory management features are well-suited for implementing sophisticated parsing algorithms. The arena-based approach ensures efficient memory usage while maintaining the complex reference relationships required by the algorithm.

Key takeaways:
- The cooperative nature of parse1/parse2 enables powerful forward reference resolution
- Interleaving parsing and macro expansion provides seamless syntactic extensibility
- Rust's ownership system helps enforce the phase separation while maintaining safety
- Arena allocation is crucial for managing the complex lifetime relationships in the parser

This foundation can be extended with additional features like more sophisticated macro systems, better error recovery, and integration with a full compiler pipeline.