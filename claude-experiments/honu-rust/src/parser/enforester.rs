use bumpalo::Bump;
use crate::{Term, TreeTerm, BracketType, ParseError, BindingInfo, Associativity, LiteralValue};
use super::BindingEnv;

pub struct Enforester<'arena> {
    pub arena: &'arena Bump,
    pub env: BindingEnv<'arena>,
}

impl<'arena> Enforester<'arena> {
    pub fn new(arena: &'arena Bump) -> Self {
        Self {
            arena,
            env: BindingEnv::new(arena),
        }
    }

    /// Parse a sequence of terms - handles both statements and expressions
    pub fn enforest_top_level(
        &mut self,
        terms: Vec<Term>,
    ) -> Result<Vec<&'arena TreeTerm<'arena>>, ParseError> {
        if terms.is_empty() {
            return Ok(vec![]);
        }

        let mut results = Vec::new();
        let mut remaining = terms;
        
        while !remaining.is_empty() {
            let (parsed, rest) = self.parse_statement_or_expression(remaining)?;
            results.push(parsed);
            remaining = rest;
        }
        
        Ok(results)
    }

    /// Parse either a statement or expression from the term stream
    fn parse_statement_or_expression(
        &mut self,
        mut terms: Vec<Term>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        if terms.is_empty() {
            return Err(ParseError::UnexpectedEndOfInput);
        }

        // Check for statement keywords first
        match terms.first() {
            Some(Term::Identifier(keyword, _)) => {
                match keyword.as_str() {
                    "if" => self.parse_if_statement(terms),
                    "while" => self.parse_while_statement(terms),
                    "for" => self.parse_for_statement(terms),
                    "function" => self.parse_function_definition(terms),
                    _ => {
                        // Not a statement keyword, parse as expression
                        self.parse_expression(terms, 0)
                    }
                }
            }
            _ => {
                // Parse as expression
                self.parse_expression(terms, 0)
            }
        }
    }

    /// Parse C-style if statement: if condition then expr else expr
    fn parse_if_statement(
        &mut self,
        mut terms: Vec<Term>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        terms.remove(0); // consume 'if'
        
        // Parse condition until 'then'
        let then_pos = terms.iter().position(|t| {
            matches!(t, Term::Identifier(name, _) if name == "then")
        }).ok_or_else(|| ParseError::ExpectedKeyword("then".to_string()))?;
        
        let condition_terms = terms.drain(..then_pos).collect();
        terms.remove(0); // consume 'then'
        
        let (condition, _) = self.parse_expression(condition_terms, 0)?;
        
        // Parse then branch until 'else' or end
        let else_pos = terms.iter().position(|t| {
            matches!(t, Term::Identifier(name, _) if name == "else")
        });
        
        let (then_branch, else_branch) = if let Some(pos) = else_pos {
            let then_terms = terms.drain(..pos).collect();
            terms.remove(0); // consume 'else'
            
            let (then_expr, _) = self.parse_expression(then_terms, 0)?;
            let (else_expr, new_remaining) = self.parse_expression(terms, 0)?;
            terms = new_remaining;
            
            (then_expr, Some(else_expr))
        } else {
            let (then_expr, remaining) = self.parse_expression(terms, 0)?;
            terms = remaining;
            (then_expr, None)
        };
        
        let if_expr = self.arena.alloc(TreeTerm::If {
            condition,
            then_branch,
            else_branch,
        });
        
        Ok((if_expr, terms))
    }

    /// Parse C-style while statement: while condition { body }
    fn parse_while_statement(
        &mut self,
        mut terms: Vec<Term>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        terms.remove(0); // consume 'while'
        
        // Find the body block
        let block_pos = terms.iter().position(|t| {
            matches!(t, Term::Bracket(BracketType::Curly, _))
        }).ok_or_else(|| ParseError::MacroExpansionError("while requires { body }".to_string()))?;
        
        let condition_terms = terms.drain(..block_pos).collect();
        let body_term = terms.remove(0);
        
        let (condition, _) = self.parse_expression(condition_terms, 0)?;
        let body = self.parse_single_term(&body_term)?;
        
        // Create while loop structure
        let while_expr = self.arena.alloc(TreeTerm::Call {
            func: self.arena.alloc(TreeTerm::Variable(
                self.arena.alloc_str("while"),
                self.env.current_scopes()
            )),
            args: vec![condition, body],
        });
        
        Ok((while_expr, terms))
    }

    /// Parse for statement: for init; condition; update { body }
    fn parse_for_statement(
        &mut self,
        mut terms: Vec<Term>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        terms.remove(0); // consume 'for'
        
        // For now, implement a simple version
        // TODO: Full for loop parsing with semicolons
        let (expr, remaining) = self.parse_expression(terms, 0)?;
        Ok((expr, remaining))
    }

    /// Parse function definition: function name(params) { body }
    fn parse_function_definition(
        &mut self,
        mut terms: Vec<Term>,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        terms.remove(0); // consume 'function'
        
        // Get function name
        let name = match terms.remove(0) {
            Term::Identifier(name, _) => name,
            _ => return Err(ParseError::ExpectedKeyword("function name".to_string())),
        };
        
        // Get parameters
        let params = match terms.remove(0) {
            Term::Bracket(BracketType::Paren, param_terms) => {
                param_terms.into_iter().map(|t| match t {
                    Term::Identifier(param, _) => Ok(param),
                    _ => Err(ParseError::MacroExpansionError("Expected parameter name".to_string())),
                }).collect::<Result<Vec<_>, _>>()?
            }
            _ => return Err(ParseError::ExpectedKeyword("parameter list".to_string())),
        };
        
        // Get body
        let body = match terms.remove(0) {
            Term::Bracket(BracketType::Curly, body_terms) => {
                // Create delayed block for forward references
                self.arena.alloc(TreeTerm::Block {
                    terms: body_terms,
                    scope_id: self.env.current_scope,
                })
            }
            _ => return Err(ParseError::ExpectedKeyword("function body".to_string())),
        };
        
        let mut param_strs: Vec<&'arena str> = Vec::new();
        for param in &params {
            param_strs.push(self.arena.alloc_str(param.as_str()));
        }
        
        let func_def = self.arena.alloc(TreeTerm::Function {
            name: self.arena.alloc_str(&name),
            params: param_strs,
            body,
        });
        
        Ok((func_def, terms))
    }

    /// Parse expressions with infix operators and macro calls
    fn parse_expression(
        &mut self,
        terms: Vec<Term>,
        min_precedence: u32,
    ) -> Result<(&'arena TreeTerm<'arena>, Vec<Term>), ParseError> {
        if terms.is_empty() {
            return Err(ParseError::UnexpectedEndOfInput);
        }

        if terms.len() == 1 {
            let expr = self.parse_single_term(&terms[0])?;
            return Ok((expr, vec![]));
        }

        // Check for macro calls first (infix macros like 'and', 'or')
        if terms.len() >= 3 {
            if let Term::Identifier(op, _) = &terms[1] {
                let scopes = self.env.current_scopes();
                if let Some(BindingInfo::Macro(transformer)) = self.env.lookup(op, &scopes) {
                    // This is an infix macro call: term1 macro term2 ...
                    let expanded = transformer.transform(terms, &self.env)?;
                    return self.parse_expression(expanded, min_precedence);
                }
            }
        }

        // Parse as infix expression with operator precedence
        // First check if this is actually an infix expression by looking for an operator in position 1
        if terms.len() >= 3 {
            if let Term::Operator(_, _) | Term::Identifier(_, _) = &terms[1] {
                let scopes = self.env.current_scopes();
                let op_name = match &terms[1] {
                    Term::Operator(op, _) => op,
                    Term::Identifier(op, _) => op,
                    _ => unreachable!(),
                };
                if let Some(BindingInfo::BinaryOp { .. }) = self.env.lookup_binary_op(op_name, &scopes) {
                    // This is an infix expression, parse the left operand
                    let mut left = self.parse_single_term(&terms[0])?;
                    let mut i = 1;
                    
                    // Continue with the rest of the infix parsing
                    while i < terms.len() {
                        let (op_name, op_span) = match &terms[i] {
                            Term::Identifier(op, span) => (op, *span),
                            Term::Operator(op, span) => (op, *span),
                            _ => break,
                        };
                        
                        let scopes = self.env.current_scopes();
                        
                        if let Some(BindingInfo::BinaryOp { precedence, assoc }) = 
                            self.env.lookup_binary_op(op_name, &scopes) {
                                
                                if *precedence < min_precedence {
                                    break;
                                }

                                let right_min_prec = match assoc {
                                    Associativity::Left => precedence + 1,
                                    Associativity::Right => *precedence,
                                };

                                if i + 1 >= terms.len() {
                                    return Err(ParseError::UnexpectedEndOfInput);
                                }

                                // Parse right side
                                let right_terms = terms[i + 1..].to_vec();
                                let (right, _) = self.parse_expression(right_terms, right_min_prec)?;

                                // Create binary operation
                                let op_str = self.arena.alloc_str(op_name);
                                left = self.arena.alloc(TreeTerm::BinaryOp {
                                    op: op_str,
                                    left,
                                    right,
                                    span: op_span,
                                });

                                i += 2; // consumed operator and right operand
                        } else {
                            // Not an operator, end of expression
                            break;
                        }
                    }

                    return Ok((left, terms[i..].to_vec()));
                }
            }
        }

        // If not an infix expression, parse as regular term
        let mut left = self.parse_single_term(&terms[0])?;
        let mut i = 1;

        while i < terms.len() {
            let (op_name, op_span) = match &terms[i] {
                Term::Identifier(op, span) => (op, *span),
                Term::Operator(op, span) => (op, *span),
                _ => break,
            };
            
            let scopes = self.env.current_scopes();
            
            if let Some(BindingInfo::BinaryOp { precedence, assoc }) = 
                self.env.lookup_binary_op(op_name, &scopes) {
                    
                    if *precedence < min_precedence {
                        break;
                    }

                    let right_min_prec = match assoc {
                        Associativity::Left => precedence + 1,
                        Associativity::Right => *precedence,
                    };

                    if i + 1 >= terms.len() {
                        return Err(ParseError::UnexpectedEndOfInput);
                    }

                    // Parse right side
                    let right_terms = terms[i + 1..].to_vec();
                    let (right, _) = self.parse_expression(right_terms, right_min_prec)?;

                    // Create binary operation
                    let op_str = self.arena.alloc_str(op_name);
                    left = self.arena.alloc(TreeTerm::BinaryOp {
                        op: op_str,
                        left,
                        right,
                        span: op_span,
                    });

                    i += 2; // consumed operator and right operand
            } else {
                // Not an operator, end of expression
                break;
            }
        }

        Ok((left, terms[i..].to_vec()))
    }

    fn parse_single_term(&mut self, term: &Term) -> Result<&'arena TreeTerm<'arena>, ParseError> {
        match term {
            Term::Identifier(name, _) => {
                let scopes = self.env.current_scopes();
                let name_str = self.arena.alloc_str(name);
                Ok(self.arena.alloc(TreeTerm::Variable(name_str, scopes)))
            }
            Term::Literal(lit) => {
                Ok(self.arena.alloc(TreeTerm::Literal(lit.clone())))
            }
            Term::Bracket(BracketType::Paren, inner_terms) => {
                if inner_terms.is_empty() {
                    Ok(self.arena.alloc(TreeTerm::Literal(LiteralValue::Nil)))
                } else if inner_terms.len() == 1 {
                    // Single term in parentheses
                    self.parse_single_term(&inner_terms[0])
                } else {
                    // Check for special constructs first, then function calls
                    match inner_terms.first() {
                        Some(Term::Identifier(name, _)) if name == "if" => {
                            // Parse if-then-else expression: (if condition then result else alternative)
                            self.parse_if_expression(inner_terms.clone())
                        }
                        Some(Term::Identifier(name, _)) if name == "loop" => {
                            // Parse loop expression: (loop condition body)
                            self.parse_loop_expression(inner_terms.clone())
                        }
                        Some(Term::Identifier(func_name, _)) | Some(Term::Operator(func_name, _)) => {
                            // Check if this is a macro first
                            let scopes = self.env.current_scopes();
                            if let Some(BindingInfo::Macro(transformer)) = self.env.lookup(func_name, &scopes) {
                                // This is a prefix macro call: (macro_name arg1 arg2 ...)
                                let expanded = transformer.transform(inner_terms.clone(), &self.env)?;
                                if expanded.len() == 1 {
                                    return self.parse_single_term(&expanded[0]);
                                } else {
                                    // Multiple terms - parse as expression
                                    let (expr, _) = self.parse_expression(expanded, 0)?;
                                    return Ok(expr);
                                }
                            }
                            
                            // Not a macro, treat as function call
                            let args: Result<Vec<_>, _> = inner_terms[1..].iter()
                                .map(|term| self.parse_single_term(term))
                                .collect();
                            let args = args?;
                            
                            let func_name_str = self.arena.alloc_str(func_name);
                            let func_var = self.arena.alloc(TreeTerm::Variable(func_name_str, scopes));
                            
                            Ok(self.arena.alloc(TreeTerm::Call {
                                func: func_var,
                                args,
                            }))
                        }
                        _ => {
                            // Parse as expression
                            let (expr, _) = self.parse_expression(inner_terms.clone(), 0)?;
                            Ok(expr)
                        }
                    }
                }
            }
            Term::Bracket(BracketType::Curly, inner_terms) => {
                // Delayed block parsing for forward references
                Ok(self.arena.alloc(TreeTerm::Block {
                    terms: inner_terms.clone(),
                    scope_id: self.env.current_scope,
                }))
            }
            Term::Bracket(BracketType::Square, inner_terms) => {
                // Array/list literal
                Ok(self.arena.alloc(TreeTerm::Block {
                    terms: inner_terms.clone(),
                    scope_id: self.env.current_scope,
                }))
            }
            Term::Operator(op, _) => {
                // Standalone operator - shouldn't happen in well-formed input
                Err(ParseError::UnexpectedOperator(op.clone()))
            }
        }
    }

    /// Parse if-then-else expression: (if condition then result else alternative)
    fn parse_if_expression(&mut self, terms: Vec<Term>) -> Result<&'arena TreeTerm<'arena>, ParseError> {
        if terms.len() < 6 {
            return Err(ParseError::MacroExpansionError(
                "if expression requires: if condition then result else alternative".to_string()
            ));
        }
        
        // Expected format: [if, condition, then, result, else, alternative]
        if !matches!(&terms[2], Term::Identifier(name, _) if name == "then") {
            return Err(ParseError::ExpectedKeyword("then".to_string()));
        }
        
        if !matches!(&terms[4], Term::Identifier(name, _) if name == "else") {
            return Err(ParseError::ExpectedKeyword("else".to_string()));
        }
        
        let condition = self.parse_single_term(&terms[1])?;
        let then_branch = self.parse_single_term(&terms[3])?;
        let else_branch = self.parse_single_term(&terms[5])?;
        
        Ok(self.arena.alloc(TreeTerm::If {
            condition,
            then_branch,
            else_branch: Some(else_branch),
        }))
    }

    /// Parse loop expression: (loop condition body)
    fn parse_loop_expression(&mut self, terms: Vec<Term>) -> Result<&'arena TreeTerm<'arena>, ParseError> {
        if terms.len() != 3 {
            return Err(ParseError::MacroExpansionError(
                "loop expression requires: loop condition body".to_string()
            ));
        }
        
        // Expected format: [loop, condition, body]
        let condition = self.parse_single_term(&terms[1])?;
        let body = self.parse_single_term(&terms[2])?;
        
        Ok(self.arena.alloc(TreeTerm::Loop {
            condition,
            body,
        }))
    }

    fn current_scopes(&self) -> crate::ScopeSet {
        self.env.current_scopes()
    }
}